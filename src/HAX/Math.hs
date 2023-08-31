{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Math where
import HAX.Tensor.Tensorial

import Data.Kind
import Data.Data (Proxy)


-- NOTE: Currently this class is just a class there a bunch of functions is putted, might split it later
class TensorOp (a :: Shape -> Type -> Type) where
  -- Automatic broadcasting
  broadcast  :: (Broadcast org map targ, Tensorial t) => a org t -> Proxy map -> a targ t
  broadcast' :: (Broadcast' org targ, Tensorial t) => a org t -> a targ t  
  
  -- TODO: Implement + - * / etc with automatic broadcasting
  prod :: (TensorProductConstraint l r p, Tensorial t) => a l t -> a r t -> a p t

  

(|#|) :: (TensorOp a, TensorProductConstraint l r p, Tensorial t) => a l t -> a r t -> a p t
(|#|) = prod
infixl 8 |#|

-- (⊗ ) :: (TensorOp a, TensorProductConstraint l r p, Tensorial t) => a l t -> a r t -> a p t
-- (⊗ ) = prod
-- infixl 8 ⊗ 
