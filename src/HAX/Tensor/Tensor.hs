{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.HostBufferSemantics
import HAX.TList (TList(..))

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
  where p = (Proxy :: Proxy a)
        shape = fromIntegral <$> shapeVal (Proxy :: Proxy s)

unity :: forall s a. (KnownShape s, Tensorial a) => Device -> IO (Tensor s a)
unity device = withArray bufferData $ \ buffer -> 
  tensorFromHostBuffer device buffer
  where bufferData = take (fromIntegral $ product $ shapeVal (Proxy :: Proxy s)) $ repeat unitElement

        
nullity :: forall s a. (KnownShape s, Tensorial a) => Device -> IO (Tensor s a)
nullity device = withArray bufferData $ \ buffer -> 
  tensorFromHostBuffer device buffer 
  where bufferData = take (fromIntegral $ product $ shapeVal (Proxy :: Proxy s)) $ repeat nullElement

splat :: forall s a. (KnownShape s, Tensorial a) => Device -> a -> IO (Tensor s a)
splat device a = withArray (take elemCount $ repeat a) $ \ a' -> 
  tensorFromHostBuffer device a'
  where elemCount = fromIntegral $ product $ shapeVal (Proxy :: Proxy s)
  





class TensorList l where 
  tensorList :: [Buffer] -> TList l
  tensorListLength :: Proxy l -> Int

instance TensorList '[] where
  tensorList [] = (:@)
  tensorList _  = error "Incorrect length list"

  tensorListLength _ = 0

instance (T s t, TensorList l) => TensorList (Tensor s t ': l) where
  tensorList (a:as) = Tensor a :+ tensorList as
  tensorList _      = error "Incorrect length list"

  tensorListLength _ = 1 + tensorListLength (Proxy :: Proxy l)





type instance K Tensor _ = ([Buffer], LoadedExecutable)
instance (T s t) => Jit Tensor (Tracer s t) (Tensor s t) where
  jit' _ _ (args, exec) = unsafePerformIO $
    Tensor . head <$> loadedExecutableExecute1Await exec args Nothing 1
  jit = error "jit should be used with a function."

instance {-# OVERLAPPING #-} (T s t) => Jit Tensor (TList '[Tracer s t]) (TList '[Tensor s t]) where
  jit' _ _ (args, exec) = tensorList $ unsafePerformIO $
    loadedExecutableExecute1Await exec args Nothing 1
  jit = error "jit should be used with a function."

instance {-# OVERLAPPABLE #-} (T s t, TensorList f', Jit Tensor (TList f) (TList f')) => Jit Tensor (TList (Tracer s t ': f)) (TList (Tensor s t ': f')) where
  jit' _ _ (args, exec) = tensorList $ unsafePerformIO $ 
    loadedExecutableExecute1Await exec args Nothing $ tensorListLength (Proxy :: Proxy (Tensor s t ': f'))
  jit = error "jit should be used with a function."

instance (KnownShape s, Tensorial t, Jit Tensor f f') => Jit Tensor (Tracer s t -> f) (Tensor s t -> f') where
  jit' pt _ (args, exec) (Tensor arg) = jit' pt pf' (args', exec)
    where args' = args ++ [arg]
          pf'   = Proxy :: Proxy f
  jit f = jit' pt pf (args, exec)
    where exec = unsafePerformIO $ compile f
          args = []
          pf   = Proxy :: Proxy (Tracer s t -> f)
          pt   = Proxy :: Proxy Tensor



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
