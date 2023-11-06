{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.AD (
  module Reverse
, traceDebugGrad
, jrgrad
)where
import HAX.Jit

import HAX.Tensor.Tensorial
import HAX.AD.Reverse as Reverse

traceDebugGrad :: (Rev (GradResult f) f ~ (a -> b), Traceable (a -> b), ReverseMode f) => f -> IO ()
traceDebugGrad x = traceDebug $ rgrad x

jrgrad :: (Rev (GradResult (a1 -> b1)) (a1 -> b1) ~ (a2 -> b2),
                TraceableElement a1, TraceableElement a2, Traceable b1,
                Traceable b2, Jit (JitTransform a1 -> JitResult b1),
                Jit (JitTransform a2 -> JitResult b2), ReverseMode (a1 -> b1)) =>
               (a1 -> b1)
               -> (JitTransform a1 -> JitResult b1,
                   JitTransform a2 -> JitResult b2)
jrgrad f = (jit f, jit $ rgrad f)

