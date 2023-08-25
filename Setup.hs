import Distribution.Simple
import Distribution.Simple.Setup

import System.Directory
import System.FilePath


baseHooks :: UserHooks
baseHooks = simpleUserHooks

userHooks :: UserHooks
userHooks = baseHooks { 
  preBuild = (\args buildFlags -> do
    let copyDst = fromFlag (buildDistPref buildFlags) </> "build" </> "libCcompops" <.> "a"
        copySrc = "deps" </> "libcompops" <.> "a"
    alreadyCopy <- doesPathExist copyDst
    if alreadyCopy then return () else copyFile copySrc copyDst 
    preBuild baseHooks args buildFlags) }



main :: IO ()
main = defaultMainWithHooks userHooks
