{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.HostBufferSemantics
-- import HAX.HList

import Data.Proxy
import Data.Kind
import Data.Primitive

import Foreign hiding (sizeOf)
import GHC.IO.Unsafe (unsafePerformIO)
import MLIR
import qualified Stablehlo.Dialect.Stablehlo as SHLO

newtype Tensor (s :: Shape) a = Tensor { getUnderlyingBuffer :: Buffer }
class Trace (t :: Shape -> Type -> Type) where
  auto :: forall s a. T s a => Tensor s a -> t s a

instance Trace Tensor where
  auto = id

instance Trace Tracer where
  auto :: forall s a. T s a => Tensor s a -> Tracer s a
  auto tensor = Tracer $ \ t0 -> do 
    buffer <- blockRunIO $ bufferToHostBuffer $ getUnderlyingBuffer tensor
    let _attr = denseElemsAttr shape buffer (Proxy :: Proxy a)
    (t0, ) <$> SHLO._ConstantOp _attr _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s a))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)

-- Pretty print tensor
instance (KnownShape s, Tensorial a, Show a, Prim a) => Show (Tensor s a) where
  show (Tensor b) = unsafePerformIO $ do
    hostbuff <- bufferToHostBuffer b
    let elemCount = sizeofByteArray hostbuff `div` elemSize
        shape     = (\ (fromIntegral -> a) -> (a - 1, a - 1)) <$> shapeVal (Proxy :: Proxy s)
    return $ fst $ formater (elemCount - 1) shape hostbuff ""
    where elemSize = sizeOf (undefined :: a)
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
  


type JitCacheTensor f = ([Buffer], LoadedExecutable, Proxy f)
type JitTensor f = (Jit Tensor f, ([Buffer], LoadedExecutable, Proxy f) ~ JitCache Tensor f)
instance T s t => Jit Tensor (Tracer s t) where
  type JitResult Tensor (Tracer s t) = Tensor s t
  type JitCache  Tensor (Tracer s t) = JitCacheTensor (Tracer s t)

  jit' (argumentStack, executable, _) = (Tensor . head . unsafePerformIO) (loadedExecutableExecute1Await executable argumentStack Nothing 1)
  jitInit _ = error "jitInit was not given a function"


--instance T s t => Jit Tensor (HList '[Tracer s t]) where
--  type JitResult Tensor (HList '[Tracer s t]) = HList '[Tensor s t]
--  type JitCache  Tensor (HList '[Tracer s t]) = JitCacheTensor (HList '[Tensor s t])
--
--  jit' (argumentStack, executable, _) = results :+ (:@)
--    where results = (Tensor . head . unsafePerformIO) (loadedExecutableExecute1Await executable argumentStack Nothing 1)
--  jitInit _ = error "jitInit was not given a function"
--
--class ListToTensorHList l where
--  l2thl :: [Buffer] -> HList l
--instance ListToTensorHList '[] where 
--  l2thl [] = (:@)
--  l2thl _  = error "Wrong length"
--instance ListToTensorHList as => ListToTensorHList (Tensor s t ': as) where
--  l2thl (a:as) = Tensor a :+ l2thl as
--  l2thl _      = error "Wrong length"
--
--type family A lhs rhs where
--  A '[] rhs = rhs 
--  A (a ': as) rhs = a ': A as rhs 
--type family B lhs rhs where
--  B (HList lhs) (HList rhs) = HList (A lhs rhs)
--
--instance (T s t, JitTensor (HList (Tracer s' t' ':ls)), HListLen (Tracer s t ': Tracer s' t' ':ls), HList k ~ JitResult Tensor (HList (Tracer s' t' ':ls))) => Jit Tensor (HList (Tracer s t ': Tracer s' t' ':ls)) where
--  type JitResult Tensor (HList (Tracer s t ': Tracer s' t' ':ls)) = B (HList '[Tensor s t]) (JitResult Tensor (HList (Tracer s' t' ': ls)))

--  type JitResult Tensor (HList (Tracer s t ': Tracer s' t' ':ls)) = HList (Tracer s t ': Tracer s' t' ':ls)
--  type JitCache  Tensor (HList (Tracer s t ': Tracer s' t' ':ls)) = JitCacheTensor (HList (Tracer s t ': Tracer s' t' ':ls))

--  jit' (argumentStack, executable, _) = l2thl results
--    where coarity = fromIntegral (hlistLen (undefined :: HList (Tracer s t ': Tracer s' t' ':ls)))
--          results = unsafePerformIO (loadedExecutableExecute1Await executable argumentStack Nothing coarity)
--  jitInit _ = error "jitInit was not given a function"



instance (T s t, JitTensor f) => Jit Tensor (Tracer s t -> f) where
  type JitResult Tensor (Tracer s t -> f) = Tensor s t -> JitResult Tensor f
  type JitCache  Tensor (Tracer s t -> f) = JitCacheTensor (Tracer s t -> f)

  jit' (argumentStack, executable, _) t = jit' (argumentStack++[getUnderlyingBuffer t], executable, Proxy :: Proxy f)
  jitInit f = ([], executable, Proxy)
    where executable = unsafePerformIO $ compile f






jit :: forall f a b. (f ~ (a -> b), JitTracer f, JitTensor f) => f -> Jit' f
jit f = jit' (jitInit f)

instance (KnownShape s, Tensorial t, Num t) => Num (Tensor s t) where
  (+) = jit f
    where f :: Tracer s t -> Tracer s t -> Tracer s t = (+)
  (-) = jit f
    where f :: Tracer s t -> Tracer s t -> Tracer s t = (-)
  (*) = jit f
    where f :: Tracer s t -> Tracer s t -> Tracer s t = (*)

  signum = jit f
    where f :: Tracer s t -> Tracer s t = signum
  abs    = jit f
    where f :: Tracer s t -> Tracer s t = abs
  negate = jit f
    where f :: Tracer s t -> Tracer s t = negate

  fromInteger = error "This is problematic"

instance (KnownShape s, Tensorial t, Fractional t) => Fractional (Tensor s t) where
  (/) = jit f
    where f :: Tracer s t -> Tracer s t -> Tracer s t = (/)
  recip = jit f
    where f :: Tracer s t -> Tracer s t = (1 /)

  fromRational = error "This is problematic"
