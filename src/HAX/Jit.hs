{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Jit where

import HAX.Tensor.Tensorial

import HAX.PjRt

import Data.Proxy
import Data.Kind

import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import MLIR

type family K (t :: Shape -> Type -> Type) f

compile :: Traceable (a -> b) => (a -> b) -> IO LoadedExecutable
compile f =
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
    return bytecode)
  where (blkM, (ins, outs)) = trace f

class Traceable f => Jit t f f' | t f -> f', f' -> t where
  jit' :: Proxy t -> Proxy f -> K t f -> f'
  jit  :: f -> f'

{- examples:
 -  a     :: Tr s0 t0 -> Tr s1 t1 -> Tr s2 t2 -> Tr s3 t3
 -  jit a :: 
 -  firstly, if the args are all tensors, then output is tensor
 -  if one or more than one tracer (or dual) is inputed, then the output is tracer (or dual)
 -  we needs to propagate the existance of 
 -  well, this is jit, it can only be either tensor or tracer 
 -  jit a :: forall j. Trace j => j s t -> F j f
 -  F Tr (_ s t -> f) = forall j. Trace j => j s t -> F Tr f
 -  F Te (_ s t -> f) = forall j. Trace 
 -  Scrap this, this is two complex
 - -}
