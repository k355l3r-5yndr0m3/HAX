{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
module HAX.AD.Reverse where
import HAX.AD.Gradient
import HAX.Tensor.Tensorial

data Reverse r s t = Reverse { primal :: r s t, cotangent :: r s t -> Gradient }

-- TODO: Restrict this to only continuous types (Float, Double, etc)
--       Discrete types don't have derivatives
instance Num (r s t) => Num (Reverse r s t) where
  (Reverse f f') + (Reverse g g') = 
    Reverse (f + g) (\ i -> f' i <+> g' i)

  (Reverse f f') - (Reverse g g') = 
    Reverse (f - g) (\ i -> f' i <+> g' (negate i))

  (Reverse f f') * (Reverse g g') = 
    Reverse (f * g) (\ i -> f' (i * g) <+> g' (i * f))

  negate (Reverse f f') = 
    Reverse (negate f) (f' . negate)

  abs    (Reverse f f') = 
    Reverse (abs f) (\ i -> f' (i * signum f))

  signum (Reverse f f') = 
    Reverse (signum f) (\ _ -> f' 0 )
  
  fromInteger a = 
    Reverse (fromInteger a) (const zero)

instance Fractional (r s t) => Fractional (Reverse r s t) where
  recip (Reverse f f') = 
    Reverse (recip f) (\ i -> f' (negate i / (f * f)))

  (Reverse f f') / (Reverse g g') = 
    Reverse (f / g) (\ i -> f' (i / g) <+> g' (negate (f / (g * g))))

  fromRational r = 
    Reverse (fromRational r) (const zero)

instance TensorOp r => TensorOp (Reverse r) where
  
