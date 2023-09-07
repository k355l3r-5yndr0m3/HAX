{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RoleAnnotations #-}
module HAX.Target where
import HAX.AD.Reverse
import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import Data.Dynamic
import Data.Coerce (coerce)

-- Target: represent the target of vmap transformation
--         Target dims r 
--         dims ++ shapeVal r is the true shape
newtype Target r s t = Target ([Integer], r s t) -- Dynamic because coerce does not work

a :: Target (Reverse Tracer) '[3, 4] Float = undefined
b :: Target (Reverse Tracer) '[3, 4, 7] Float = coerce a
