{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
module HAX.Jit where

import HAX.Tensor.Tensorial

import HAX.PjRt
import HAX.Utils

import Control.Exception
import Data.Proxy
import Data.Bifunctor

import MLIR

import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import GHC.IO.Unsafe

{-# NOINLINE compile #-}
compile :: Traceable (a -> b) => (a -> b) -> (Int, LoadedExecutable)
compile f = unsafePerformIO $ (length outs, ) <$> (
  clientCompile client =<< runContextM (do
    loadDialect_ Func.dialect
    loadDialect_ SHLO.dialect
    m <- moduleOp $ do 
      Func._FuncOp "main"
                   (TypeAttr $ FunctionType ins outs)
                   Nothing Nothing Nothing $ do 
        bb0 <- blockGet ins 
        blockDef bb0 $ do 
          _out <- blkM
          Func._ReturnOp _out
    bytecode <- writeByteCode (moduleGetOperation m) 
    moduleDestroy m 
    return bytecode))
  where (blkM, (ins, outs)) = trace f

class JitReify f where 
  jitReify :: Annotated [Buffer] f -> (f, [Buffer])
  jitUnreify :: Annotated [Buffer] a -> f -> Annotated [Buffer] b

instance (JitReify a, JitReify b) => JitReify (a <+> b) where
  jitReify (Annotated outs) = (a :+: b, bs)
    where (a, as) = jitReify (Annotated outs :: Annotated [Buffer] a)
          (b, bs) = jitReify (Annotated as   :: Annotated [Buffer] b)
  jitUnreify args (a :+: b) = jitUnreify (jitUnreify args a) b

type JitData f = (Annotated [Buffer] f, (Int, LoadedExecutable))
class Jit f where
  jit' :: JitData f -> f

instance {-# OVERLAPPABLE #-} JitReify j => Jit j where
  jit' (Annotated args, (nout, program)) = assert (null leftover) result 
    where outputs :: Annotated [Buffer] j = Annotated . unsafePerformIO $ loadedExecutableExecute1Await program args Nothing nout
          (result, leftover) = jitReify outputs

instance {-# OVERLAPPING #-} (JitReify i, Jit f) => Jit (i -> f) where
  jit' jitData i = jit' (first (`jitUnreify` i) jitData)

-- JitTransform should never receive a function
type family JitTransform a
type instance JitTransform (Proxy a) = Proxy a 
type instance JitTransform (a <+> b) = JitTransform a <+> JitTransform b
type family JitResult f where
  JitResult (a ->  b) = JitTransform a -> JitResult b
  JitResult a         = JitTransform a

{-# NOINLINE jit #-}
jit :: forall a b f f'. (Traceable f, Jit f', f' ~ JitResult f, f ~ (a -> b)) => f -> f' 
jit f = jit' (Annotated [] :: Annotated [Buffer] f', compile f)
