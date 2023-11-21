{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.AD (
  module Reverse
, jrgrad
)where
import HAX.Tensor.Tensor

import HAX.AD.Reverse as Reverse

jrgrad :: (Jit f, Jit (GradF f), Grad f) => f -> (JitF f, JitF (GradF f))
jrgrad f = (jit f, jit $ grad f)
