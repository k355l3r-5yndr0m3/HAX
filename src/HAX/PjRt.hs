{-# LANGUAGE UnliftedFFITypes #-}
module HAX.PjRt where

import HAX.PjRt.HostBufferSemantics (HostBufferSemantics)
import HAX.PjRt.BufferType (BufferType)
import HAX.PjRt.Plugin (ShapeInfo(..))

import qualified HAX.PjRt.Plugin           as C
import qualified HAX.PjRt.Client           as C
import qualified HAX.PjRt.Event            as C
import qualified HAX.PjRt.LoadedExecutable as C
import qualified HAX.PjRt.Buffer           as C

import GHC.IO.Unsafe (unsafePerformIO)
import Foreign
import MLIR.IR (ByteCode)
import Data.Primitive.ByteArray (ByteArray)
import Control.Monad (forM)
import Data.Unique

import Paths_HAX

-- Global singleton
-- TODO: Implement using a nullity type class
{-# NOINLINE api #-}
api :: Ptr C.Api
api = unsafePerformIO $ C.loadPjRtPlugin =<< getDataFileName "deps/libPJRTPlugin.so"

{-# NOINLINE client #-}
client :: Client
client = unsafePerformIO clientCreate

-- TODO: Implement a scheme where the devices are choosen at runtime for 
--       efficiency. 
{-# NOINLINE defaultDevice #-}
defaultDevice :: Device
defaultDevice = head $ clientAddressableDevices client

-- New types
newtype Client           = Client           (Ptr C.Client)

-- The IO is there to keep client alive
newtype Device           = Device           (Ptr C.Device)
newtype Event            = Event            (ForeignPtr C.Event)
newtype Buffer           = Buffer           (ForeignPtr C.Buffer)
newtype LoadedExecutable = LoadedExecutable (ForeignPtr C.LoadedExecutable)

-- Note: some of these functions will probably not be used 
--       as their lower level counter part will be used instead
-- Client 
clientCreate :: IO Client
clientCreate = Client <$> C.clientCreate api

clientDestroy :: Client -> IO ()
clientDestroy (Client c) = C.clientDestroy api c

clientPlatformName :: Client -> IO String
clientPlatformName (Client c) = C.clientPlatformName api c

clientCompile :: Client -> ByteCode -> IO LoadedExecutable
clientCompile (Client c) bytecode = do
  i <- newUnique
  putStrLn $ "Compiling " ++ show (hashUnique i) -- Tibit of code to check for recompilation, debugging only
  executable <- newForeignPtrEnv C.loadedExecutableDestroy__ptr api =<< C.clientCompile api c bytecode 
  return $ LoadedExecutable executable

-- The output of this function should not change unless there is a feature of xla that I'm not aware of
clientAddressableDevices :: Client -> [Device]
clientAddressableDevices (Client c) = unsafePerformIO $ do
  devices <- C.clientAddressableDevices api c
  return $ Device <$> devices 
  
clientBufferFromHostBuffer :: Client -> Ptr a -> BufferType -> ShapeInfo -> HostBufferSemantics -> Device -> IO (Event, Buffer)
clientBufferFromHostBuffer (Client c) hostBuffer bufferType shapeInfo hostBufferSemantics (Device d) = do
  (eventPtr, bufferPtr) <- C.clientBufferFromHostBuffer api c hostBuffer bufferType shapeInfo hostBufferSemantics d
  event <- newForeignPtrEnv C.eventDestroy__ptr api eventPtr
  buffer <- newForeignPtrEnv C.bufferDestroy__ptr api bufferPtr
  return (Event event, Buffer buffer)

-- Buffer
bufferToHostBuffer :: Buffer -> IO ByteArray
bufferToHostBuffer (Buffer b) = 
  withForeignPtr b $ \ b' -> 
    C.bufferToHostBuffer api b'


-- Event
eventAwait :: Event -> IO ()
eventAwait (Event e) = do 
  withForeignPtr e $ \ e' -> 
    C.eventAwait api e'

-- Loaded executable
loadedExecutableExecute1Await :: LoadedExecutable -> [Buffer] -> Maybe Device -> Int -> IO [Buffer]
loadedExecutableExecute1Await (LoadedExecutable e) argList runOnDevice numOutputs = do 
  withForeignPtr e $ \ e' -> 
    withForeignPtrList ((\ (Buffer b) -> b) <$> argList) $ \argList' -> 
      let device = 
            case runOnDevice of
              Nothing         -> nullPtr
              Just (Device d) -> d
      in  do 
        outputs <- C.loadedExecutableExecute1Await api e' argList' device numOutputs
        forM outputs (fmap Buffer . newForeignPtrEnv C.bufferDestroy__ptr api)
  where withForeignPtrList :: [ForeignPtr a] -> ([Ptr a] -> IO b) -> IO b
        withForeignPtrList []     f = f []
        withForeignPtrList (p:ps) f = 
          withForeignPtr p $ \ p' -> 
            let f' ps' = f (p' : ps')
            in  withForeignPtrList ps f'
