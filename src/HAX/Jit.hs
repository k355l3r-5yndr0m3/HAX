{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module HAX.Jit where

import HAX.Tensor.Tensorial

import HAX.PjRt

import Data.Kind

import MLIR

import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

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


type Jit' f = forall t. Jit t f => JitResult t f
class Traceable f => Jit (t :: Shape -> Type -> Type) (f :: Type) where
  type JitResult t f = r | r -> t f
  type JitCache  t f = c | c -> t f

  jit' :: JitCache t f -> JitResult t f
  jitInit :: f -> JitCache t f

  jitReify :: [Buffer] -> (JitResult t f, [Buffer])

jit :: f ~ (a -> b) => f -> Jit' f
jit f = jit' (jitInit f)









