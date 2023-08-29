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
