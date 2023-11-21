{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE LambdaCase #-}
module HAX.Utils where
import Prelude hiding (lookup)
import Data.IntMap.Strict

import GHC.Generics (Generic)
import GHC.StableName 

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

newtype StableCache i = StableCache (IntMap [(forall a. StableName a -> Bool, i)])
cacheLookup :: StableName a -> StableCache i -> Maybe i
cacheLookup name@(hashStableName -> hash) (StableCache cache) = 
  cache !? hash >>= search
  where search :: [(forall b. StableName b -> Bool, i)] -> Maybe i
        search = \case 
          []                -> Nothing
          (verify, item):ls 
            | verify name   -> Just item
            | otherwise     -> search ls
cacheInsert :: forall a i. StableName a -> i -> StableCache i -> StableCache i
cacheInsert name@(hashStableName -> hash) item (StableCache cache) = StableCache $
  insertWith insertion hash [(eqStableName name, item)] cache
  where insertion :: e -> [(forall n. StableName n -> Bool, i)] -> [(forall n. StableName n -> Bool, i)]
        insertion _ table = (eqStableName name, item):clear table
        clear :: [(forall n. StableName n -> Bool, i)] -> [(forall n. StableName n -> Bool, i)]
        clear = \case 
          []            -> []
          (v, i):ls 
            | v name    -> ls
            | otherwise -> (v, i):clear ls

emptyCache :: StableCache i
emptyCache = StableCache empty
