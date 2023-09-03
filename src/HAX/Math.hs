{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Math where
import HAX.Tensor.Tensorial

import Data.Kind
import Data.Data (Proxy)


-- NOTE: Currently this class is just a class there a bunch of functions is putted, might split it later

-- (⊗ ) :: (TensorOp a, TensorProductConstraint l r p, Tensorial t) => a l t -> a r t -> a p t
-- (⊗ ) = prod
-- infixl 8 ⊗ 
