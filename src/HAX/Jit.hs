{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ImpredicativeTypes #-}
module HAX.Jit where
import HAX.PjRt

import MLIR

import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import Data.IORef
import GHC.IO.Unsafe (unsafePerformIO)
import HAX.Tensor.Tracer (StableNameHashTable (StableNameHashTable))

import Data.IntMap.Strict


-- Currently, this can only support types where the number of tensors contained within that type is known at compile time
-- One fix, use fix length list, easier to implement, slight hasel for the user
-- Two implement a structure that contained the dynamic size
-- class JitReify f where 
--   jitReify   :: Annotated [Buffer] f -> (f, [Buffer])
--   jitUnreify :: Annotated [Buffer] a -> f -> Annotated [Buffer] b
-- 
-- instance (JitReify a, JitReify b) => JitReify (a <&> b) where
--   jitReify (Annotated outs) = (a :&: b, bs)
--     where (a, as) = jitReify (Annotated outs :: Annotated [Buffer] a)
--           (b, bs) = jitReify (Annotated as   :: Annotated [Buffer] b)
--   jitUnreify args (a :&: b) = jitUnreify (jitUnreify args a) b
-- 
-- type JitData f = (Annotated [Buffer] f, (Int, LoadedExecutable))
-- class Jit f where
--   jit' :: JitData f -> f
-- 
-- instance {-# OVERLAPPABLE #-} JitReify j => Jit j where
--   jit' (Annotated args, (nout, program)) = assert (null leftover) result 
--     where outputs :: Annotated [Buffer] j = Annotated . unsafePerformIO $ loadedExecutableExecute1Await program args Nothing nout
--           (result, leftover) = jitReify outputs
-- 
-- instance {-# OVERLAPPING #-} (JitReify i, Jit f) => Jit (i -> f) where
--   jit' jitData i = jit' (first (`jitUnreify` i) jitData)
-- 
-- -- JitTransform should never receive a function
-- type family JitTransform a
-- type instance JitTransform (a -> b)  = TypeError (Text "Jit should not receive high order function.")
-- type instance JitTransform (Proxy a) = Proxy a 
-- type instance JitTransform (a <&> b) = JitTransform a <&> JitTransform b
-- type instance JitTransform [a]     = [JitTransform a] -- Dynamically sized, needed to be recorded
-- type family JitResult f where
--   JitResult (a ->  b) = JitTransform a -> JitResult b
--   JitResult a         = JitTransform a
-- 
-- {-# NOINLINE jit #-}
-- type J f f' = (Traceable f, Jit f', f' ~ JitResult f)
-- jit :: J f f' => f -> f' 
-- jit f = jit' (Annotated [] :: Annotated [Buffer] f', compile f)

{-# NOINLINE compilationCounter #-}
compilationCounter :: IORef Int
compilationCounter = unsafePerformIO (newIORef 0)

compile :: ([AnyType], StableNameHashTable Value -> BlockM (StableNameHashTable Value, [Value]), [AnyType]) -> IO LoadedExecutable
compile (ins, main, outs) = atomicModifyIORef compilationCounter (\i -> (i + 1, undefined)) >> (
  clientCompile client =<< runContextM (do
    loadDialect_ Func.dialect
    loadDialect_ SHLO.dialect
    m <- moduleOp $ do 
      Func._FuncOp "main"
                   (TypeAttr $ FunctionType ins outs)
                   Nothing Nothing Nothing $ do 
        bb0 <- blockGet ins 
        blockDef bb0 $ do 
          (_, _out) <- main $ StableNameHashTable empty
          Func._ReturnOp _out
    bytecode <- writeByteCode (moduleGetOperation m) 
    moduleDestroy m 
    return bytecode))

compileDebug :: ([AnyType], StableNameHashTable Value -> BlockM (StableNameHashTable Value, [Value]), [AnyType]) -> IO LoadedExecutable
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
          (_, _out) <- main $ StableNameHashTable empty
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
