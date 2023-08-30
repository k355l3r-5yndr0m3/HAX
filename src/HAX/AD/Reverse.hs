{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
module HAX.AD.Reverse where
import HAX.AD.Gradient

data Reverse t = Reverse { primal :: t, cotangent :: t -> Gradient }

instance Num t => Num (Reverse t) where
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

instance Fractional t => Fractional (Reverse t) where
  recip (Reverse f f') = 
    Reverse (recip f) (\ i -> f' (negate i / (f * f)))

  (Reverse f f') / (Reverse g g') = 
    Reverse (f / g) (\ i -> f' (i / g) <+> g' (negate (f / (g * g))))

  fromRational r = 
    Reverse (fromRational r) (const zero)
