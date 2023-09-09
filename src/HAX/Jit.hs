{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
module HAX.Jit where

import HAX.Tensor.Tensorial

import HAX.PjRt
import HAX.Utils

import Control.Exception

import MLIR

import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import GHC.IO.Unsafe

compile :: Traceable (a -> b) => (a -> b) -> IO (Int, LoadedExecutable)
compile f = (length outs, ) <$> (
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


type JitData f = (Annotated [Buffer] f, (Int, LoadedExecutable))
class JitReify f where 
  jitReify :: Annotated [Buffer] f -> (f, [Buffer])

class Jit f where
  jit' :: JitData f -> f

instance (JitReify a, JitReify b) => JitReify (a <+> b) where
  jitReify (Annotated outs) = (a :+: b, bs)
    where (a, as) = jitReify (Annotated outs :: Annotated [Buffer] a)
          (b, bs) = jitReify (Annotated as   :: Annotated [Buffer] b)

instance (JitReify a, JitReify b) => Jit (a <+> b) where
  jit' (Annotated args, (nout, program)) = assert (null leftover) result
    where outputs :: Annotated [Buffer] (a <+> b) = Annotated . unsafePerformIO $ loadedExecutableExecute1Await program args Nothing nout
          (result, leftover) = jitReify outputs

type family JitTransform a
type family JitResult f where
  JitResult (a ->  b) = JitTransform a -> JitResult b
  JitResult (a <+> b) = JitResult a <+> JitResult b
  JitResult a         = JitTransform a

jit :: forall a b f f'. (Traceable f, Jit f', f' ~ JitResult f, f ~ (a -> b)) => f -> f' 
jit f = jit' (Annotated [] :: Annotated [Buffer] f', unsafePerformIO $ compile f)
