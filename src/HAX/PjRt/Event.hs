{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnliftedFFITypes #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.PjRt.Event where
import HAX.PjRt.Plugin 

import Foreign

foreign import ccall unsafe "event_destroy"
  eventDestroy :: Ptr Api -> Ptr Event -> IO ()
foreign import ccall unsafe "event_await"
  eventAwait :: Ptr Api -> Ptr Event -> IO ()
foreign import ccall unsafe "event_wait_then_destroy"
  eventWaitThenDestroy :: Ptr Api -> Ptr Event -> IO ()
foreign import ccall unsafe "&event_destroy"
  eventDestroy__ptr :: FinalizerEnvPtr Api Event
