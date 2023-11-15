{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.AD (
  module Reverse
-- , traceDebugGrad
, jrgrad
)where
import HAX.Tensor.Tensor

import HAX.AD.Reverse as Reverse

-- traceDebugGrad :: (Rev (GradResult f) f ~ (a -> b), Traceable (a -> b), ReverseMode f) => f -> IO ()
-- traceDebugGrad x = traceDebug $ rgrad x

-- jrgrad :: (Jit f, Jit (Rev (GradResult f) f), ReverseMode f) => f -> (JitF f, JitF (Rev (GradResult f) f)) 
jrgrad :: (Jit f, Jit (GradF f), Grad f) => f -> (JitF f, JitF (GradF f))
jrgrad f = (jit f, jit $ rgrad f)
