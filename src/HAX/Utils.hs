module HAX.Utils where

newtype Annotated a b = Annotated a

data a <+> b = a :+: b deriving Show
instance (Num a, Num b) => Num (a <+> b) where
  alhs :+: blhs + arhs :+: brhs = 
    (alhs + arhs) :+: (blhs + brhs)

  alhs :+: blhs - arhs :+: brhs = 
    (alhs - arhs) :+: (blhs - brhs)

  alhs :+: blhs * arhs :+: brhs = 
    (alhs * arhs) :+: (blhs * brhs)

  abs (a :+: b) = abs a :+: abs b
  negate (a :+: b) = negate a :+: negate b
  signum (a :+: b) = signum a :+: signum b

  fromInteger i = fromInteger i :+: fromInteger i


