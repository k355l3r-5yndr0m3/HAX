import Distribution.Simple
import Distribution.Simple.Setup

import System.Directory
import System.FilePath

baseHooks :: UserHooks
baseHooks = simpleUserHooks

userHooks :: UserHooks
userHooks = baseHooks { 
  preBuild = (\args buildFlags -> do
    preBuild baseHooks args buildFlags) }

main :: IO ()
main = defaultMainWithHooks userHooks
