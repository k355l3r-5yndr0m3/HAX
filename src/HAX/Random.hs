module HAX.Random where
import HAX.Tensor.Tensor
import HAX.Tensor.Tensorial

import HAX.PjRt

import Control.Monad 

import Data.Proxy

import Foreign

import System.Random
import System.Random.Stateful

import GHC.IO.Unsafe

-- TODO: Implement polymorphism
tensorUniformRM :: forall s t g m. (StatefulGen g m, UniformRange (StorageType t), T s t) => (t, t) -> g -> m (Tensor s t)
tensorUniformRM r g = do 
  entropy <- replicateM nelem (uniformRM r' g)
  return $ unsafePerformIO (tensorFromHostBufferGC defaultDevice =<< newArray entropy)
  where nelem = fromIntegral $ product $ shapeVal (Proxy :: Proxy s)
        r'    = 
          let (a, b) = r 
          in  (fromHaskell a, fromHaskell b)

tensorUniformR :: (RandomGen g, UniformRange (StorageType t), T s t) => (t, t) -> g -> (Tensor s t, g)
tensorUniformR r g = runStateGen g (tensorUniformRM r)

-- TODO: Implement random for <+>
