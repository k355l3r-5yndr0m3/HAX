{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnliftedFFITypes #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.PjRt.LoadedExecutable (
  loadedExecutableDestroy 
, loadedExecutableExecute1Await
-- , loadedExecutableExecute1AwaitForeignPtrs
, loadedExecutableDestroy__ptr
) where
import HAX.PjRt.Plugin 

import Foreign
import Foreign.C

foreign import ccall unsafe "loaded_executable_destroy"
  loadedExecutableDestroy :: Ptr Api -> Ptr LoadedExecutable -> IO ()
foreign import ccall unsafe "&loaded_executable_destroy"
  loadedExecutableDestroy__ptr :: FinalizerEnvPtr Api LoadedExecutable

foreign import ccall unsafe "loaded_executable_execute__1_await"
  c__loadedExecutableExecute__1Await :: Ptr Api -> Ptr LoadedExecutable -> Ptr ExecuteOptions -> Ptr (Ptr Buffer) -> CSize -> Ptr (Ptr Buffer) -> Ptr Device -> IO ()
loadedExecutableExecute1Await :: Ptr Api -> Ptr LoadedExecutable -> [Ptr Buffer] -> Ptr Device -> Int -> IO [Ptr Buffer]
loadedExecutableExecute1Await api executable args device numOutputs = 
  withArrayLen args $ \ (fromIntegral -> numArgs) argList ->
  allocaArray numOutputs $ \ outputList -> do 
    c__loadedExecutableExecute__1Await api executable nullPtr argList numArgs outputList device
    peekArray numOutputs outputList

-- loadedExecutableExecute1AwaitForeignPtrs :: Ptr Api -> Ptr LoadedExecutable -> [Ptr Buffer] -> Maybe Device -> Int -> IO [ForeignPtr Buffer]
-- loadedExecutableExecute1AwaitForeignPtrs api exec buffers device numOutputs =  
--   withForeignPtrList buffers $ \ (fmap Buffer -> buffers') -> 
--       loadedExecutableExecute1Await api exec buffers' device numOutputs
--   where withForeignPtrList :: [ForeignPtr a] -> ([Ptr a] -> IO b) -> IO b
--         withForeignPtrList []     f = f []
--         withForeignPtrList (p:ps) f = 
--           withForeignPtr p $ \ p' -> 
--             let f' ps' = f (p' : ps')
--             in  withForeignPtrList ps f'
