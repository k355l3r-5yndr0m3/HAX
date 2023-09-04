{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.AD.Numerical where
import Data.Proxy

class Delta t where
  type Scalar t
  scalarDelta :: Proxy t -> Scalar t
  deltas :: [t]


