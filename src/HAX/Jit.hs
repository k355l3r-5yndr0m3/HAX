{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ImpredicativeTypes #-}
module HAX.Jit where
import HAX.PjRt
import HAX.Utils

import MLIR

import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import Data.IORef
import GHC.IO.Unsafe (unsafePerformIO)

import Data.IntMap.Strict

{-# NOINLINE compilationCounter #-}
compilationCounter :: IORef Int
compilationCounter = unsafePerformIO (newIORef 0)

{-# NOINLINE compile #-}
compile :: ([AnyType], StableCache Value -> BlockM (StableCache Value, [Value]), [AnyType]) -> LoadedExecutable
compile = unsafePerformIO . compile'

compile' :: ([AnyType], StableCache Value -> BlockM (StableCache Value, [Value]), [AnyType]) -> IO LoadedExecutable
compile' (ins, main, outs) = atomicModifyIORef compilationCounter (\i -> (i + 1, undefined)) >> (
  clientCompile client =<< runContextM (do
    loadDialect_ Func.dialect
    loadDialect_ SHLO.dialect
    m <- moduleOp $ do 
      Func._FuncOp "main"
                   (TypeAttr $ FunctionType ins outs)
                   Nothing Nothing Nothing $ do 
        bb0 <- blockGet ins 
        blockDef bb0 $ do 
          (_, _out) <- main $ StableCache empty
          Func._ReturnOp _out
    bytecode <- writeByteCode (moduleGetOperation m) 
    moduleDestroy m 
    return bytecode))

compileDebug :: ([AnyType], StableCache Value -> BlockM (StableCache Value, [Value]), [AnyType]) -> IO LoadedExecutable
compileDebug (ins, main, outs) = atomicModifyIORef compilationCounter (\i -> (i + 1, undefined)) >> (
  clientCompile client =<< runContextM (do
    loadDialect_ Func.dialect
    loadDialect_ SHLO.dialect
    m <- moduleOp $ do 
      Func._FuncOp "main"
                   (TypeAttr $ FunctionType ins outs)
                   Nothing Nothing Nothing $ do 
        bb0 <- blockGet ins 
        blockDef bb0 $ do 
          (_, _out) <- main $ StableCache empty
          Func._ReturnOp _out
    moduleDump m
    bytecode <- writeByteCode (moduleGetOperation m) 
    moduleDestroy m 
    return bytecode))

getNumCompilations :: IO Int
getNumCompilations = readIORef compilationCounter

echoNumCompilations :: IO ()
echoNumCompilations = do 
  n <- getNumCompilations
  putStrLn $ "Number of compilation performed: " ++ show n
