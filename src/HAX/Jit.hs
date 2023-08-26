{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FunctionalDependencies #-}
module HAX.Jit where

import HAX.Tensor.Typeclass
import HAX.Tensor.Shape

import HAX.PjRt

import Data.Proxy
import Data.Kind (Type)

import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import MLIR hiding (Type)

type family K (t :: Shape -> Type -> Type) f

-- TODO: make a newtype for ForeignPtr of loaded executable
compile :: Traceable (a -> b) => (a -> b) -> IO LoadedExecutable
compile f =
  clientCompile client =<< withContext (do 
    loadDialect_ Func.dialect
    loadDialect_ SHLO.dialect
    m <- moduleOp $ do 
      Func._FuncOp (stringAttr "main")
                   (typeAttr $ functionType ins outs)
                   Nothing Nothing Nothing $ do 
        bb0 <- addBlock ins 
        defBlock bb0 $ do 
          _out <- blkM
          Func._ReturnOp _out
    bytecode <- moduleWriteBytecode m 
    moduleDestroy m 
    return bytecode)
  where (blkM, (ins, outs)) = trace f


class (Trace t, Traceable f) => Jit t f f' | t f -> f', f' -> t where
  jit' :: Proxy t -> Proxy f -> K t f -> f'
  jit  :: f -> f'





