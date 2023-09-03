{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Math 
import HAX.Jit
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.HostBufferSemantics
import HAX.Utils

import Data.Proxy
import Data.Kind
import Data.Primitive

import Foreign hiding (sizeOf)
import GHC.IO.Unsafe (unsafePerformIO)
import MLIR
import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Control.Exception (assert)

newtype Tensor (s :: Shape) a = Tensor { getUnderlyingBuffer :: Buffer }
class Trace (t :: Shape -> Type -> Type) where
  auto :: forall s a. T s a => Tensor s a -> t s a

instance Trace Tensor where
  auto = id

instance Trace Tracer where
  auto :: forall s a. T s a => Tensor s a -> Tracer s a
  auto tensor = Tracer 0 $ \ t0 _ -> do 
    buffer <- blockRunIO $ bufferToHostBuffer $ getUnderlyingBuffer tensor
    let _attr = denseElemsAttr shape buffer (Proxy :: Proxy a)
    (t0, ) <$> SHLO._ConstantOp _attr _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s a))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)

-- Pretty print tensor
instance (T s a, Show a, Prim a) => Show (Tensor s a) where
  show (Tensor b) = unsafePerformIO $ do
    hostbuff <- bufferToHostBuffer b
    let elemCount = sizeofByteArray hostbuff `div` elemSize
        shape     = (\ (fromIntegral -> a) -> (a - 1, a - 1)) <$> shapeVal (Proxy :: Proxy s)
    return $ fst $ formater (elemCount - 1) shape hostbuff ""
    where elemSize = staticSizeOf (Proxy :: Proxy a)
          formater :: Int -> [(Int, Int)] -> ByteArray -> String -> (String, Int)
          formater offs [] buf s = 
            let a :: a = indexByteArray buf offs
            in  (show a ++ s, offs - 1)
          formater offs ((idx, ext):ies) buf s
            | idx == 0  =
              let (s', offs') = formater offs ies buf (',':s)
              in  ('[':s', offs')
            | otherwise = 
              let c = if idx == ext then ']' else ','
                  (s', offs') = formater offs ies buf (c:s)
              in  formater offs' ((idx - 1, ext):ies) buf s'




tensorFromHostBuffer :: forall s a. (KnownShape s, Tensorial a) => Device -> Ptr a -> IO (Tensor s a)
tensorFromHostBuffer device buffer = Tensor <$> do 
  (e, b) <- clientBufferFromHostBuffer client buffer (pjrtBufferType p) (Shape shape) kImmutableOnlyDuringCall device
  eventAwait e
  return b
  where p = Proxy :: Proxy a
        shape = fromIntegral <$> shapeVal (Proxy :: Proxy s)

unity :: forall s a. (KnownShape s, Tensorial a) => Device -> IO (Tensor s a)
unity device = withArray bufferData $ \ buffer -> 
  tensorFromHostBuffer device buffer
  where bufferData = replicate (fromIntegral $ product $ shapeVal (Proxy :: Proxy s)) unitElement

        
nullity :: forall s a. (KnownShape s, Tensorial a) => Device -> IO (Tensor s a)
nullity device = withArray bufferData $ \ buffer -> 
  tensorFromHostBuffer device buffer 
  where bufferData = replicate (fromIntegral $ product $ shapeVal (Proxy :: Proxy s)) nullElement

splat :: forall s a. (KnownShape s, Tensorial a) => Device -> a -> IO (Tensor s a)
splat device a = withArray (replicate elemCount a) $ \ a' -> 
  tensorFromHostBuffer device a'
  where elemCount = fromIntegral $ product $ shapeVal (Proxy :: Proxy s)
  


type JitCacheTensor f = ([Buffer], LoadedExecutable, Annotated Int f)
type JitTensor f = (Jit Tensor f, JitCacheTensor f ~ JitCache Tensor f)
instance Jit Tensor (Proxy (Tracer s t)) where
  type JitResult Tensor (Proxy (Tracer s t)) = Proxy (Tensor s t)
  type JitCache  Tensor (Proxy (Tracer s t)) = JitCacheTensor (Proxy (Tracer s t))
  
  jit' _ = Proxy
  jitInit _ = error "jitInit was not given a function"

  jitReify l = (Proxy, l)

instance T s t => Jit Tensor (Tracer s t) where
  type JitResult Tensor (Tracer s t) = Tensor s t
  type JitCache  Tensor (Tracer s t) = JitCacheTensor (Tracer s t)

  jit' (argumentStack, executable, _) = assert (null excess) output
    where (output, excess) = (jitReify . unsafePerformIO) (loadedExecutableExecute1Await executable argumentStack Nothing 1)

  jitInit _ = error "jitInit was not given a function"

  jitReify (a:as) = (Tensor a, as)
  jitReify _      = error "Computation does not produce all demanded results"

instance (JitTensor a, JitTensor b) => Jit Tensor (a <+> b) where
  type JitResult Tensor (a <+> b) = JitResult Tensor a <+> JitResult Tensor b
  type JitCache  Tensor (a <+> b) = JitCacheTensor (a <+> b)

  jit' (argumentStack, executable, Annotated coarity) = assert (null excess) output
    where (output, excess) = (jitReify . unsafePerformIO) (loadedExecutableExecute1Await executable argumentStack Nothing coarity)
  
  jitInit _ = error "jitInit was not given a function"
  jitReify r0 = (a :+: b, r2)
    where (a, r1) = jitReify r0
          (b, r2) = jitReify r1
  

instance (T s t, JitTensor f) => Jit Tensor (Tracer s t -> f) where
  type JitResult Tensor (Tracer s t -> f) = Tensor s t -> JitResult Tensor f
  type JitCache  Tensor (Tracer s t -> f) = JitCacheTensor (Tracer s t -> f)

  jit' (argumentStack, executable, Annotated coarity) t = jit' (argumentStack++[getUnderlyingBuffer t], executable, Annotated coarity)

  jitInit f = ([], executable, Annotated coarity)
    where (coarity, executable) = unsafePerformIO $ compile f

  jitReify = error "Should not be possible"


jit :: forall f a b. (f ~ (a -> b), JitTracer f, JitTensor f) => f -> Jit' f
jit f = jit' (jitInit f)

instance (KnownShape s, Tensorial t, Num t) => Num (Tensor s t) where
  (+) = jit (+)
  (-) = jit (-)
  (*) = jit (*)

  signum = jit signum
  abs    = jit abs
  negate = jit negate

  fromInteger = error "This is problematic"

instance (KnownShape s, Tensorial t, Fractional t) => Fractional (Tensor s t) where
  (/) = jit (/)
  recip = jit recip

  fromRational = error "This is problematic"

instance TensorOp Tensor where
  broadcast :: forall t (org :: Shape) (map :: Shape) (targ :: Shape). (Broadcast org map targ, Tensorial t) => Tensor org t -> Proxy map -> Tensor targ t
  broadcast i p = jit f i
    where f :: Tracer org t -> Tracer targ t 
          f = (`broadcast` p)
  broadcast' = jit broadcast'

  prod :: forall l r p t. (TensorProductConstraint l r p, Tensorial t) => Tensor l t -> Tensor r t -> Tensor p t
  prod = jit prod
