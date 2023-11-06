{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD.Gradient where

import Data.Dynamic
import Data.List
import Data.Proxy
import Type.Reflection

import Foreign.C

-- TODO: Optimize
newtype Gradient = Gradient [(CIntPtr, Dynamic)]

(<+>) :: Gradient -> Gradient -> Gradient
(Gradient lhs) <+> (Gradient rhs) = Gradient (lhs ++ rhs)

zero :: Gradient
zero = Gradient []

nograd :: a -> Gradient 
nograd = const zero

independent :: Typeable t => CIntPtr -> t -> Gradient
independent idx val = Gradient [(idx, toDyn val)]

reifyGrad :: (Typeable t, Num t) => CIntPtr -> Gradient -> (t, Gradient)
reifyGrad idx (Gradient gradient) = (grad, other)
  where (fmap snd -> g, Gradient -> other) = partition ((== idx) . fst) gradient
        grad = sum $ fmap fromDyn' g

type Cotangent t = (Typeable t, Num t)

fromDyn' :: forall a. Typeable a => Dynamic -> a
fromDyn' d = 
  case fromDynamic d of 
    Just d' -> d'
    Nothing -> error ("Not of expected type (Actual: " ++ show (dynTypeRep d) ++ ")\n\
                      \ (Expected: " ++ show (someTypeRep (Proxy :: Proxy a)) ++ ")")
