{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
module HAX.AD.Reverse where
import HAX.Tensor.Tensorial
import HAX.Tensor.Transform
import HAX.AD.Gradient


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

instance Floating (r s t) => Floating (Reverse r s t) where
  pi = Reverse pi (const zero)
  exp (Reverse f f') = Reverse (exp f) (\ i -> f' (i * exp f))
  log (Reverse f f') = Reverse (log f) (\ i -> f' (i / f))
  sqrt (Reverse f f') = Reverse (sqrt f) (\ i -> f' (i / (2 * sqrt f)))
  (Reverse f f') ** (Reverse g g') = 
    Reverse (f ** g) (\ i -> f' (i * g * (f ** (g - 1))) <+> g' (i * (f ** g) * log f))
  sin (Reverse f f') = Reverse (sin f) (\ i -> f' (i * cos f))
  cos (Reverse f f') = Reverse (cos f) (\ i -> f' (negate (i * sin f)))
  tanh (Reverse f f') = Reverse (tanh f) (\ i -> f' (i * (1 - tanh f ** 2)))

instance TensorOp r => TensorOp (Reverse r) where




