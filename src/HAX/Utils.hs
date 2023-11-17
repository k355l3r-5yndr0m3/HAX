{-# LANGUAGE DataKinds #-}
module HAX.Utils where
import GHC.Generics (Generic)

newtype Annotated a b = Annotated a
data a <&> b = a :&: b deriving (Show, Generic)
infixl 8 :&:, <&>
instance (Num a, Num b) => Num (a <&> b) where
  alhs :&: blhs + arhs :&: brhs = 
    (alhs + arhs) :&: (blhs + brhs)

  alhs :&: blhs - arhs :&: brhs = 
    (alhs - arhs) :&: (blhs - brhs)

  alhs :&: blhs * arhs :&: brhs = 
    (alhs * arhs) :&: (blhs * brhs)

  abs (a :&: b) = abs a :&: abs b
  negate (a :&: b) = negate a :&: negate b
  signum (a :&: b) = signum a :&: signum b

  fromInteger i = fromInteger i :&: fromInteger i

instance (Fractional a, Fractional b) => Fractional (a <&> b) where
  alhs :&: blhs / arhs :&: brhs = 
    (alhs / arhs) :&: (blhs / brhs)
  recip (lhs :&: rhs) = recip lhs :&: recip rhs
  fromRational r = fromRational r :&: fromRational r


