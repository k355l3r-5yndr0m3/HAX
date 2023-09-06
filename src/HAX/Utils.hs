module HAX.Utils where

newtype Annotated a b = Annotated a

data a <+> b = a :+: b deriving Show
