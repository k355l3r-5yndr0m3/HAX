{-# LANGUAGE ForeignFunctionInterface #-}
module HAX.PjRt.Plugin (
  Api
, Client
, Buffer
, Device
, Event
, LoadedExecutable

, MemoryLayout 
, ExecuteOptions
, ShapeInfo(..)

, loadPjRtPlugin
) where
import Foreign (Ptr(), Int64)
import Foreign.C (CString, withCString)


data Api 
data Client
data Buffer
data Device
data Event
data LoadedExecutable

-- newtype Api              = Api              (Ptr Api)
-- newtype Client           = Client           (Ptr Client)
-- newtype Buffer           = Buffer           (Ptr Buffer) deriving (Storable)
-- newtype Device           = Device           (Ptr Device) deriving (Storable)
-- newtype Event            = Event            (Ptr Event) deriving (Storable)
-- newtype LoadedExecutable = LoadedExecutable (Ptr ())


data MemoryLayout
data ExecuteOptions
data ShapeInfo = Shape [Int64]
               | ShapeAndByteStrides [(Int64, Int64)] 


foreign import ccall unsafe "load_pjrt_plugin" 
  loadPjRtPlugin' :: CString -> IO (Ptr Api)
loadPjRtPlugin :: String -> IO (Ptr Api)
loadPjRtPlugin = (`withCString` loadPjRtPlugin')
