{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE NoStarIsType #-}
module HAX.NN.Model where
import Data.Proxy

import HAX.Tensor
import HAX.AD.Numerical
import HAX.AD

import GHC.TypeLits
import GHC.Generics

class Model p a b | p a -> b {- , p b -> a -} where
  feed :: p -> a -> b

-- Combination
data a >> b = a :>: b deriving (Generic, Show)
infixr 9 >>, :>:
instance (NGradIn f, NGradIn g) => NGradIn (f >> g)

data a !! b = a :!: b deriving (Generic, Show)
infixr 9 !!, :!:
instance (NGradIn f, NGradIn g) => NGradIn (f !! g)
instance (Jit' a, Jit' b) => Jit' (a >> b) where
  type JitT (a >> b)   = JitT a >> JitT b
  type JitC (a >> b) c = JitC' (Rep (a >> b)) c
instance (Jit' a, Jit' b) => Jit (a >> b) where
instance (GradIn a, GradIn b) => GradIn (a >> b) where
  type GradI (a >> b) = GradI a >> GradI b
instance (Model f a i, Model g i b) => Model (f >> g) a b where
  feed (f :>: g) = feed g . feed f

instance (Jit' a, Jit' b) => Jit' (a !! b) where
  type JitT (a !! b) = JitT a !! JitT b
  type JitC (a !! b) c = JitC' (Rep (a !! b)) c
instance (Jit' a, Jit' b) => Jit (a !! b) where
instance (GradIn a, GradIn b) => GradIn (a !! b) where
  type GradI (a !! b) = GradI a !! GradI b
instance (Model f a b, Model g a b, Num b) => Model (f !! g) a b where 
  feed (f :!: g) x = feed f x + feed g x

newtype Residual l = Residual l deriving (Generic, Show)
instance NGradIn f => NGradIn (Residual f)
instance Jit' f    => Jit' (Residual f) where
  type JitT (Residual f)   = Residual (JitT f)
  type JitC (Residual f) b = JitC' (Rep (Residual f)) b
instance Jit' f    => Jit  (Residual f) where
instance (Model f a a, Num a) => Model (Residual f) a a where
  feed (Residual f) a = a + feed f a

-- Dense 
newtype Dense (r :: Z) t (i :: Nat) (o :: Nat) = Dense (r [i, o] t, r '[o] t) deriving Generic
instance (KnownNat i, KnownNat o) => NGradIn (Dense Tensor Float i o)
instance (Show (r [i, o] t), Show (r '[o] t)) => Show (Dense r t i o) where
  show (Dense (weights, biases)) = "Dense " ++ show weights ++ " " ++ show biases
instance (GNT r, KnownNat i, KnownNat o, Tensorial t) => GradIn (Dense r t i o) where
  type GradI (Dense r t i o) = Dense (Ins r) t i o
instance (JNT r, KnownNat i, KnownNat o, Tensorial t) => Jit' (Dense r t i o) where
  type JitT (Dense r t i o)   = Dense Tensor t i o
  type JitC (Dense r t i o) b = JitC' (Rep (Dense r t i o)) b
instance (JNT r, KnownNat i, KnownNat o, Tensorial t) => Jit (Dense r t i o) where

instance (TensorOp r, KnownNat i, KnownNat o, Tensorial t, Num t) => Model (Dense r t i o) (r '[i] t) (r '[o] t) where
  feed (Dense (weights, biases)) feats = biases `unsafePairwiseAdd` linearMap weights feats

-- Reshaping
data Reshape (i :: Shape) (o :: Shape) = Reshape deriving (Generic, Show)
instance NGradIn (Reshape i o)
instance (KnownShape i, KnownShape o) => GradIn (Reshape i o) where
  type GradI (Reshape i o) = Reshape i o
instance (KnownShape i, KnownShape o) => Jit' (Reshape i o) where
  type JitT (Reshape i o) = Reshape i o
  type JitC (Reshape i o) b = JitC' (Rep (Reshape i o)) b
instance (KnownShape i, KnownShape o) => Jit (Reshape i o) where
instance (TensorOp r, Tensorial t, KnownShape i, KnownShape o, Reshapable i o) => Model (Reshape i o) (r i t) (r o t) where 
  feed _ = reshape

-- Nonlinear activation functions
data Sigmoid = Sigmoid deriving (Generic, Show)
instance NGradIn Sigmoid
instance Jit' Sigmoid where
  type JitT Sigmoid   = Sigmoid
  type JitC Sigmoid b = JitC' (Rep Sigmoid) b
instance Jit Sigmoid where
instance GradIn Sigmoid where
  type GradI Sigmoid = Sigmoid 
instance (TensorOp r, T s t, Floating t) => Model Sigmoid (r s t) (r s t) where
  feed _ = sigmoid

data ReLU = ReLU deriving (Generic, Show)
instance NGradIn ReLU
instance Jit' ReLU where
  type JitT ReLU = ReLU
  type JitC ReLU b = JitC' (Rep ReLU) b
instance Jit ReLU where
instance GradIn ReLU where
  type GradI ReLU = ReLU
instance (TensorOp r, T s t, Num t, Ord t) => Model ReLU (r s t) (r s t) where
  feed _ = relu

data Softmax = Softmax deriving (Generic, Show)
instance NGradIn Softmax
instance Jit' Softmax where
  type JitT Softmax = Softmax
  type JitC Softmax b = JitC' (Rep Softmax) b
instance Jit Softmax where
instance GradIn Softmax where
  type GradI Softmax = Softmax
instance (TensorOp r, T s t, Floating t) => Model Softmax (r s t) (r s t) where 
  feed _ = softmax'

-- Convolution
type family (:+) (a :: [x]) (b :: x) :: [x] where
  '[]       :+ b = '[b]
  (a ': as) :+ b = a ': (as :+ b)
instance (KnownNat i, KnownShape s, KnownNat o, KnownShape (s :+ o)) => NGradIn (Convolute Tensor Float i s o) where
instance (GNT r, Tensorial t, KnownNat i, KnownShape s, KnownNat o, KnownShape (s :+ o)) => GradIn (Convolute r t i s o) where
  type GradI (Convolute r t i s o) = Convolute (Ins r) t i s o
instance (JNT r, Tensorial t, KnownNat i, KnownShape s, KnownNat o, KnownShape (s :+ o)) => Jit' (Convolute r t i s o) where
  type JitT (Convolute r t i s o)   = Convolute Tensor t i s o
  type JitC (Convolute r t i s o) b = JitC' (Rep (Convolute r t i s o)) b
instance (JNT r, Tensorial t, KnownNat i, KnownShape s, KnownNat o, KnownShape (s :+ o)) => Jit (Convolute r t i s o) where
data Convolute (r :: Z) t (i :: Nat) (k :: Shape) (o :: Nat) = Convolute (r (i:(k :+ o)) t) (r '[o] t) deriving (Generic)
instance (Tensorial t, KnownNat i, KnownShape s, KnownNat o, KnownShape (s :+ o)) => Show (Convolute Tensor t i s o) where
  show (Convolute weights biases) = "Convolute " ++ show weights ++ " " ++ show biases
type family Add (a :: Nat) (b :: Nat) :: Nat where
  Add a b = a + b
instance (TensorOp r, Num t, Tensorial t, KnownNat i, KnownNat o) => Model (Convolute r t i '[] o) (r '[i] t) (r '[o] t) where -- degenerate case
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

-- Transformer
data MultiHeadAttentionGeneral (r :: Z) t (head :: Nat) (key :: Nat) (query :: Nat) (key' :: Nat) (val :: Nat) (val' :: Nat) (out :: Nat) = MultiHeadAttentionGeneral (r [query, head, key'] t) (r [key, head, key'] t) (r [val, head, val'] t) (r [head, val', out] t) deriving Generic
instance (N head, N key, N query, N key', N val, N val', N out) => NGradIn (MultiHeadAttentionGeneral Tensor Float head key query key' val val' out)
instance (GNT r, Tensorial t, N head, N key, N query, N key', N val, N val', N out) => GradIn (MultiHeadAttentionGeneral r t head key query key' val val' out) where
  type GradI (MultiHeadAttentionGeneral r t head key query key' val val' out) = MultiHeadAttentionGeneral (Ins r) t head key query key' val val' out
instance (JNT r, Tensorial t, N head, N key, N query, N key', N val, N val', N out) => Jit' (MultiHeadAttentionGeneral r t head key query key' val val' out) where
  type JitT (MultiHeadAttentionGeneral r t head key query key' val val' out)   = MultiHeadAttentionGeneral Tensor t head key query key' val val' out
  type JitC (MultiHeadAttentionGeneral r t head key query key' val val' out) b = JitC' (Rep (MultiHeadAttentionGeneral r t head key query key' val val' out)) b
instance (JNT r, Tensorial t, N head, N key, N query, N key', N val, N val', N out) => Jit (MultiHeadAttentionGeneral r t head key query key' val val' out) where
instance (TensorOp r, Tensorial t, Floating t, N head, N key, N query, N key', N val, N val', N out, N queries, N keys) =>
          Model (MultiHeadAttentionGeneral r t head key query key' val val' out) (r [queries, query] t, r [keys, key] t, r [keys, val] t) (r [queries, out] t) where
  feed (MultiHeadAttentionGeneral wq wk wv wo) (queries, keys, values) = mha wq wk wv wo queries keys values
newtype MultiHeadAttention (r :: Z) t (head :: Nat) (dmodel :: Nat) = MultiHeadAttention (MultiHeadAttentionGeneral r t head dmodel dmodel (dmodel `Div` head) dmodel (dmodel `Div` head) dmodel) deriving Generic
instance (N head, N dmodel, N (dmodel `Div` head)) => NGradIn (MultiHeadAttention Tensor Float head dmodel)
instance (GNT r, Tensorial t, N head, N dmodel, N (dmodel `Div` head)) => GradIn (MultiHeadAttention r t head dmodel) where
  type GradI (MultiHeadAttention r t head dmodel) = MultiHeadAttention (Ins r) t head dmodel
instance (JNT r, Tensorial t, N head, N dmodel, N (dmodel `Div` head)) => Jit' (MultiHeadAttention r t head dmodel) where
  type JitT (MultiHeadAttention r t head dmodel)   = MultiHeadAttention Tensor t head dmodel
  type JitC (MultiHeadAttention r t head dmodel) b = JitC' (Rep (MultiHeadAttention r t head dmodel)) b
instance (JNT r, Tensorial t, N head, N dmodel, N (dmodel `Div` head)) => Jit (MultiHeadAttention r t head dmodel) where
instance (TensorOp r, Tensorial t, Floating t, N head, N dmodel, Mod dmodel head ~ 0, N queries, N keys, KnownNat (dmodel `Div` head)) => 
          Model (MultiHeadAttention r t head dmodel) (r [queries, dmodel] t, r [keys, dmodel] t, r [keys, dmodel] t) (r [queries, dmodel] t) where
  feed (MultiHeadAttention m) = feed m
