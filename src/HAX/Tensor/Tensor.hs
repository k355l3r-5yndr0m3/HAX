{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit hiding (jit)
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.Utils

import Control.Exception

import Data.Proxy
import Data.Kind
import Data.Primitive hiding (newArray)
import Data.Int

import Foreign

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.IsList

import MLIR
import qualified Stablehlo.Dialect.Stablehlo as SHLO

newtype Tensor (s :: Shape) a = Tensor { getUnderlyingBuffer :: Buffer }
debugTensorShape :: Tensor s t -> IO [Int64]
debugTensorShape = bufferDimensions . getUnderlyingBuffer

-- TODO: Consider removing the class Trace, it is currently doing nothing
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
-- TODO: Fix, because this is just bad
--       consider using bytestring
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
              let (s', offs') = formater offs ies buf ((if idx == ext then ']' else ','):s)
              in  ('[':s', offs')
            | otherwise = 
              let c = if idx == ext then ']' else ','
                  (s', offs') = formater offs ies buf (c:s)
              in  formater offs' ((idx - 1, ext):ies) buf s'



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
  fromList l = unsafePerformIO $ tensorFromHostBufferGC defaultDevice =<< l'
    where p = Proxy :: Proxy s
          l' = newArray $ flatten p $ regularize p (nullElement :: t) l
  toList = error "TODO: Implement"

tensorFromHostBufferGC :: forall s t. T s t => Device -> Ptr t -> IO (Tensor s t)
tensorFromHostBufferGC device hostBuffer = Tensor <$>
  clientBufferFromHostBufferGC client hostBuffer (pjrtBufferType p) (Shape shape) device
  where p :: Proxy t = Proxy
        shape = fromIntegral <$> shapeVal (Proxy :: Proxy s)

tensorSplat :: forall s t. T s t => Device -> t -> IO (Tensor s t)
tensorSplat device a = do
  content <- mallocArray elemCount
  sequence_ [pokeElemOff content i a | i <- [0..elemCount - 1]]
  tensorFromHostBufferGC device content
  where elemCount = fromIntegral $ product $ shapeVal (Proxy :: Proxy s)

instance JitReify (Tensor s t) where
  jitReify (Annotated (a:as)) = (Tensor a, as)
  jitReify (Annotated [])     = error "Program did not produced enough outputs"

instance Jit (Tensor s t) where 
  jit' (Annotated args, (nout, program)) = assert (null leftover) result
    where outputs :: Annotated [Buffer] (Tensor s t) = Annotated . unsafePerformIO $ loadedExecutableExecute1Await program args Nothing nout
          (result, leftover) = jitReify outputs

instance Jit f => Jit (Tensor s t -> f) where
  jit' (Annotated args, prog) (Tensor b) = jit' (Annotated (args ++ [b]), prog)

type instance JitTransform (Tracer s t) = Tensor s t

-- Implement a jit the convert from function of tracers to tensors 
type family JitTracerTransform f
type instance JitTracerTransform (Tensor s t) = Tracer s t
type family JitTracer f where 
  JitTracer (a ->  b) = JitTracerTransform a -> JitTracer b
  JitTracer (a <+> b) = JitTracer a <+> JitTracer b
  JitTracer a         = JitTracerTransform a

jit :: forall a b f f'. (f ~ (a -> b), Traceable f, f' ~ JitResult f, Jit f', f ~ JitTracer f') => f -> f'
jit f = jit' jitData
  where jitData = (Annotated [] :: Annotated [Buffer] f', ) . unsafePerformIO $ compile f

instance (T s t, Num t) => Num (Tensor s t) where
  (+) = jit (+)
  (-) = jit (-)
  (*) = jit (*)

  signum = jit signum
  abs    = jit abs
  negate = jit negate

  fromInteger = unsafePerformIO . tensorSplat defaultDevice . fromIntegral

instance (KnownShape s, Tensorial t, Fractional t) => Fractional (Tensor s t) where
  (/) = jit (/)
  recip = jit recip

  fromRational = unsafePerformIO . tensorSplat defaultDevice . fromRational

instance Tensorial t => TensorOp Tensor t where
  unsafeBroadcast operand dims = jit (`unsafeBroadcast` dims) operand
  unsafeDotGeneral lhs rhs attr = jit (\ _lhs _rhs -> unsafeDotGeneral _lhs _rhs attr) lhs rhs
  unsafeTranspose operand perm = jit (`unsafeTranspose` perm) operand

  splat a = unsafePerformIO $ tensorSplat defaultDevice a
  unsafeReduceAdd operand axies = jit (`unsafeReduceAdd` axies) operand
  unsafeReduceMul operand axies = jit (`unsafeReduceMul` axies) operand
