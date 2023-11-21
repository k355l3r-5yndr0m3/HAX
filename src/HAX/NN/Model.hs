{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
module HAX.NN.Model where
import Data.Proxy

import HAX.Tensor
import HAX.AD.Numerical
import HAX.AD

import GHC.TypeLits
import GHC.Generics

class Model p a b | p a -> b, p b -> a where
  feed :: p -> a -> b
instance (Jitter a, Jitter b) => Jitter (a >> b) where
  type JitT (a >> b) = JitT a >> JitT b
instance (Jitter a, Jitter b) => Jit (a >> b) where
  type JitF (a >> b) = JitT (a >> b)
instance (GradIn a, GradIn b) => GradIn (a >> b) where
  type GradI (a >> b) = GradI a >> GradI b
instance (Model f a i, Model g i b) => Model (f >> g) a b where
  feed (f :>: g) = feed g . feed f

instance (Jitter a, Jitter b) => Jitter (a !! b) where
  type JitT (a !! b) = JitT a !! JitT b
instance (Jitter a, Jitter b) => Jit (a !! b) where
  type JitF (a !! b) = JitT (a !! b)
instance (GradIn a, GradIn b) => GradIn (a !! b) where
  type GradI (a !! b) = GradI a !! GradI b
instance (Model f a b, Model g a b, Num b) => Model (f !! g) a b where 
  feed (f :!: g) x = feed f x + feed g x

data a >> b = a :>: b deriving (Generic, Show)
infixr 9 >>, :>:
instance (NGradIn f, NGradIn g) => NGradIn (f >> g)

data a !! b = a :!: b deriving (Generic, Show)
infixr 9 !!, :!:
instance (NGradIn f, NGradIn g) => NGradIn (f !! g)

-- Dense 
newtype Dense (r :: Z) t (i :: Nat) (o :: Nat) = Dense (r [i, o] t, r '[o] t) deriving Generic
instance (KnownNat i, KnownNat o) => NGradIn (Dense Tensor Float i o)
instance (Show (r [i, o] t), Show (r '[o] t)) => Show (Dense r t i o) where
  show (Dense (weights, biases)) = "Dense " ++ show weights ++ " " ++ show biases
instance (GNT r, KnownNat i, KnownNat o, Tensorial t) => GradIn (Dense r t i o) where
  type GradI (Dense r t i o) = Dense (Ins r) t i o
instance (JNT r, KnownNat i, KnownNat o, Tensorial t) => Jitter (Dense r t i o) where
  type JitT (Dense r t i o) = Dense Tensor t i o
instance (JNT r, KnownNat i, KnownNat o, Tensorial t) => Jit (Dense r t i o) where
  type JitF (Dense r t i o) = JitT (Dense r t i o)

instance (TensorOp r, KnownNat i, KnownNat o, Tensorial t, Num t) => Model (Dense r t i o) (r '[i] t) (r '[o] t) where
  feed (Dense (weights, biases)) feats = biases `unsafePairwiseAdd` linearMap weights feats

-- Reshaping
data Reshape (i :: Shape) (o :: Shape) = Reshape deriving (Generic, Show)
instance NGradIn (Reshape i o)
instance (KnownShape i, KnownShape o) => GradIn (Reshape i o) where
  type GradI (Reshape i o) = Reshape i o
instance (KnownShape i, KnownShape o) => Jitter (Reshape i o) where
  type JitT (Reshape i o) = Reshape i o
instance (KnownShape i, KnownShape o) => Jit (Reshape i o) where
  type JitF (Reshape i o) = JitT (Reshape i o)
instance (TensorOp r, Tensorial t, KnownShape i, KnownShape o, Reshapable i o) => Model (Reshape i o) (r i t) (r o t) where 
  feed _ = reshape

-- Nonlinear activation functions
data Sigmoid = Sigmoid deriving (Generic, Show)
instance NGradIn Sigmoid
instance Jitter Sigmoid where
  type JitT Sigmoid = Sigmoid
instance Jit Sigmoid where
  type JitF Sigmoid = JitT Sigmoid
instance GradIn Sigmoid where
  type GradI Sigmoid = Sigmoid 
instance (T s t, Floating (r s t)) => Model Sigmoid (r s t) (r s t) where
  feed _ = sigmoid

data ReLU = ReLU deriving (Generic, Show)
instance NGradIn ReLU
instance Jitter ReLU where
  type JitT ReLU = ReLU
instance Jit ReLU where
  type JitF ReLU = JitT ReLU
instance GradIn ReLU where
  type GradI ReLU = ReLU
instance (TensorOp r, T s t, Num t) => Model ReLU (r s t) (r s t) where
  feed _ = relu

data Softmax = Softmax deriving (Generic, Show)
instance NGradIn Softmax
instance Jitter Softmax where
  type JitT Softmax = Softmax
instance Jit Softmax where
  type JitF Softmax = JitT Softmax
instance GradIn Softmax where
  type GradI Softmax = Softmax
instance (TensorOp r, T s t, Floating t) => Model Softmax (r s t) (r s t) where 
  feed _ = softmax

type family (:+) (a :: [x]) (b :: x) :: [x] where
  '[]       :+ b = '[b]
  (a ': as) :+ b = a ': (as :+ b)

data Convolute (r :: Z) t (i :: Nat) (k :: Shape) (o :: Nat) = Convolute (r (i:(k :+ o)) t) (r '[o] t)
type family Add (a :: Nat) (b :: Nat) :: Nat where
  Add a b = a + b
instance (TensorOp r, Num t, Tensorial t, KnownNat i, KnownNat o) => Model (Convolute r t i '[] o) (r '[i] t) (r '[o] t) where
  feed (Convolute weights biases) = unsafePairwiseAdd biases . linearMap weights

instance (TensorOp r, Tensorial t, KnownNat i, KnownNat o, c ~ a + d - 1, d ~ c - a + 1, KnownNat c, KnownNat d, KnownNat a, 
          Model (Convolute r t i as o) (r ins t) (r out t), Num t,
          KnownShape as, KnownShape ins, KnownShape out, KnownShape (as :+ o)) => Model (Convolute r t i (a ': as) o) (r (c ': ins) t) (r (d ': out) t) where
  feed (Convolute kernel biases) input = result + biases'
    where result  = unsafeReshape result'
          result' = unsafeConvolution input' kernel :: r (1 ': d ': out) t
          input'  = unsafeReshape input :: r (1 ': c ': ins) t
          biases' = unsafeBroadcast biases [rank - 1]
          rank    = shapeRank (Proxy :: Proxy (d ': out))
