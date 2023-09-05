{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE MagicHash #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.HostBufferSemantics
import HAX.Utils
import HAX.AD.Numerical

import Control.Exception

import Data.Proxy
import Data.Kind
import Data.Primitive

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.IsList

import MLIR
import qualified Stablehlo.Dialect.Stablehlo as SHLO

newtype Tensor (s :: Shape) a = Tensor { getBuffer :: Buffer }
class Trace (t :: Shape -> Type -> Type) where
  auto :: forall s a. T s a => Tensor s a -> t s a

instance Trace Tensor where
  auto = id

instance Trace Tracer where
  auto :: forall s a. T s a => Tensor s a -> Tracer s a
  auto tensor = Tracer 0 $ \ t0 _ -> do 
    buffer <- blockRunIO $ bufferToHostBuffer $ getBuffer tensor
    let _attr = denseElemsAttr shape buffer (Proxy :: Proxy a)
    (t0, ) <$> SHLO._ConstantOp _attr _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s a))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)

-- Pretty print tensor
-- TODO: Extend this to work with other layout
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


tensorFromHostBuffer :: forall s t. T s t => Device -> PrimArray t -> IO (Tensor s t)
tensorFromHostBuffer device buffer = assert (isPrimArrayPinned buffer) Tensor <$> do 
  (e, b) <- clientBufferFromHostBuffer client (primArrayToByteArray buffer) (pjrtBufferType p) (Shape shape) kImmutableOnlyDuringCall device
  eventAwait e
  return b
  where p = Proxy :: Proxy t
        shape = fromIntegral <$> shapeVal (Proxy :: Proxy s)
        primArrayToByteArray :: PrimArray a -> ByteArray
        primArrayToByteArray (PrimArray a#) = ByteArray a#

tensorToHostBuffer :: forall s t. T s t => Tensor s t -> IO (PrimArray t)
tensorToHostBuffer tensor = do 
  ByteArray byteArray# <- bufferToHostBuffer (getBuffer tensor)
  return $ PrimArray byteArray#

unity :: forall s a. T s a => Device -> IO (Tensor s a)
unity = (`splat` unitElement)
        
nullity :: forall s a. T s a => Device -> IO (Tensor s a)
nullity = (`splat` nullElement)

splat :: forall s a. T s a => Device -> a -> IO (Tensor s a)
splat device a = tensorFromHostBuffer device $ primArrayFromList $ replicate (fromIntegral $ product $ shapeVal (Proxy :: Proxy s)) a

-- TODO: Figure out how to extend this to n-rank
-- instance T '[r0] t => IsList (Tensor '[r0] t) where
--   type Item (Tensor '[r0] t) = t
--   fromList content = unsafePerformIO $ tensorFromHostBuffer defaultDevice (primArrayFromList padded)
--     where shape  = fromInteger <$> shapeVal (Proxy :: Proxy '[r0])
--           padded = take (last shape) (content ++ repeat nullElement)
--   toList tensor = unsafePerformIO $ primArrayToList <$> tensorToHostBuffer tensor
-- 
-- instance T '[r1, r0] t => IsList (Tensor '[r1, r0] t) where
--   type Item (Tensor '[r1, r0] t) = [t]
--   fromList content = unsafePerformIO $ tensorFromHostBuffer defaultDevice (primArrayFromList (concat padded))
--     where shape  = fromInteger <$> shapeVal (Proxy :: Proxy '[r1, r0])
--           padded = take (head shape) (((\ c -> take (last shape) (c ++ repeat nullElement)) <$> content) ++ repeat (replicate (last shape) nullElement))
--   toList = error "TODO: Implement"

resizeList :: Int -> a -> [a] -> [a]
resizeList len pad list
  | len > 0   = 
    case list of
      []     -> replicate len pad
      (a:as) -> a:resizeList (len - 1) pad as
  | otherwise = []

class T s t => ListToTensor s t where
  type Padding s t
  regularize :: Proxy s -> t -> [Padding s t] -> [Padding s t] 
  padding    :: Proxy s -> t -> Padding s t
  flatten    :: Proxy s -> [Padding s t] -> [t]

instance T '[r0] t => ListToTensor '[r0] t where
  type Padding '[r0] t = t
  regularize _ = resizeList n 
    where n = fromInteger $ shapeValHead (Proxy :: Proxy '[r0])
  padding _ i = i
  flatten _ = id

instance (T (a ': as ': ass) t, ListToTensor (as ': ass) t) => ListToTensor (a ': as ': ass) t where
  type Padding  (a ': as ': ass) t = [Padding (as ': ass) t]
  regularize p t l = resizeList n (padding p t) l'
    where n  = fromInteger $ shapeValHead (Proxy :: Proxy (a ': as ': ass))
          l' = fmap (regularize (Proxy :: Proxy (as ': ass)) t) l
  padding _ i = replicate n (padding (Proxy :: Proxy (as ': ass)) i)
    where n = fromInteger $ shapeValHead (Proxy :: Proxy (as ': ass))
  flatten _ = concatMap (flatten (Proxy :: Proxy (as ': ass)))

instance ListToTensor s t => IsList (Tensor s t) where
  type Item (Tensor s t) = Padding s t
  fromList l = unsafePerformIO $ tensorFromHostBuffer defaultDevice l'
    where p = Proxy :: Proxy s
          l' = primArrayFromList $ flatten p $ regularize p (nullElement :: t) l
  toList = error "TODO: Implement"

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
    where (output, excess) = (jitReify . unsafePerformIO) (loadedExecutableExecute1Await executable argumentStack (Just defaultDevice) 1)

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

  jit' (argumentStack, executable, Annotated coarity) t = jit' (argumentStack++[getBuffer t], executable, Annotated coarity)

  jitInit f = ([], executable, Annotated coarity)
    where (coarity, executable) = unsafePerformIO $ compile f

  jitReify = error "Should not be possible"


jit :: forall f a b. (f ~ (a -> b), JitTracer f, JitTensor f) => f -> Jit' f
jit f = jit' (jitInit f)

instance (T s t, Num t) => Num (Tensor s t) where
  (+) = jit (+)
  (-) = jit (-)
  (*) = jit (*)

  signum = jit signum
  abs    = jit abs
  negate = jit negate

  fromInteger (fromInteger -> value :: t) = unsafePerformIO $ splat defaultDevice value

instance (T s t, Fractional t) => Fractional (Tensor s t) where
  (/) = jit (/)
  recip = jit recip

  fromRational (fromRational -> value :: t) = unsafePerformIO $ splat defaultDevice value

instance (T s t, Floating t) => Floating (Tensor s t) where
  pi = unsafePerformIO $ splat defaultDevice (pi :: t)
  exp = jit exp
  log = jit log
  sqrt = jit sqrt
  (**) = jit (**)  
  sin = jit sin
  cos = jit cos
  tanh = jit tanh
  asin = error "No equivalent stablehlo operation"
  acos = error "No equivalent stablehlo operation"
  atan = error "No equivalent stablehlo operation"
  sinh = error "No equivalent stablehlo operation"
  cosh = error "No equivalent stablehlo operation"
  asinh = error "No equivalent stablehlo operation"
  acosh = error "No equivalent stablehlo operation"
  atanh = error "No equivalent stablehlo operation"

instance TensorOp Tensor where
  broadcast :: forall t (org :: Shape) (map :: Shape) (targ :: Shape). (Broadcast org map targ, Tensorial t) => Tensor org t -> Proxy map -> Tensor targ t
  broadcast i p = jit f i
    where f :: Tracer org t -> Tracer targ t 
          f = (`broadcast` p)
  broadcast' = jit broadcast'

  reduceAdditive t r = jit (`reduceAdditive` r) t
  reduceMultiplicative t r = jit (`reduceMultiplicative` r) t

  prod = jit prod
  dot = jit dot

-- instance (T s t, Fractional t) => Delta (Tensor s t) where
--   type Scalar (Tensor s t) = t
--   scalarDelta _ = 0.0001 -- TODO: Choose another constant, one that is less arbitrary
