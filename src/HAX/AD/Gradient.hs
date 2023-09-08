{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD.Gradient where

import Data.Dynamic
import Data.Maybe
import Data.List
import Foreign.C

-- TODO: Optimize
newtype Gradient = Gradient [(CIntPtr, Dynamic)]

(<+>) :: Gradient -> Gradient -> Gradient
(Gradient lhs) <+> (Gradient rhs) = Gradient (lhs ++ rhs)

zero :: Gradient
zero = Gradient []

independent :: Typeable t => CIntPtr -> t -> Gradient
independent idx val = Gradient [(idx, toDyn val)]

reifyGrad :: (Typeable t, Num t) => CIntPtr -> Gradient -> (t, Gradient)
reifyGrad idx (Gradient gradient) = (grad, other)
  where (fmap snd -> g, Gradient -> other) = partition ((== idx) . fst) gradient
        grad = sum $ fmap (fromJust . fromDynamic) g

type Cotangent t = (Typeable t, Num t)
