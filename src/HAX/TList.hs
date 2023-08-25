{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
module HAX.TList where
import Data.Kind (Type)
import Data.Tuple (Solo (MkSolo))
import GHC.TypeLits
import Data.Data (Proxy (Proxy))

data TList (t :: [Type]) where
  (:@) :: TList '[]
  (:+) :: a -> TList as -> TList (a ': as)
infixr 5 :+, :@

class TListTuple l t | l -> t, t -> l where
  tlist2tup :: l -> t
  tup2tlist :: t -> l

instance TListTuple (TList '[]) () where
  tlist2tup _ = ()
  tup2tlist _ = (:@)

instance TListTuple (TList '[a]) (Solo a) where
  tlist2tup (a :+ (:@)) = MkSolo a
  tup2tlist (MkSolo a)  = a :+ (:@)

instance TListTuple (TList '[a, b]) (a, b) where
  tlist2tup (a :+ b :+ (:@)) = (a, b)
  tup2tlist (a, b) = (a :+ b :+ (:@))

instance TListTuple (TList '[a, b, c]) (a, b, c) where
  tlist2tup (a :+ b :+ c :+ (:@)) = (a, b, c)
  tup2tlist (a, b, c) = (a :+ b :+ c :+ (:@))


class TListConcat a b where
  type Concat a b :: [Type]
  (++:) :: TList a -> TList b -> TList (Concat a b)

instance TListConcat '[] b where
  type Concat '[] b = b
  _ ++: b = b

instance TListConcat as b => TListConcat (a ': as) b where
  type Concat (a ': as) b = a ': Concat as b
  (a :+ as) ++: b = a :+ (as ++: b)


class TListReplaceAt (i :: Nat) (l :: [Type]) t | l i -> t where
  replaceAt :: Proxy i -> TList l -> t -> TList l

instance TListReplaceAt 0 (a ': as) a where
  replaceAt _ (_ :+ as) a = a :+ as

instance TListReplaceAt (i - 1) l t => TListReplaceAt i (a ': l) t where
  replaceAt _ (l :+ ls) a = l :+ replaceAt p ls a 
    where p :: Proxy (i - 1) = Proxy


instance Num (TList '[]) where
  _ + _ = (:@)
  _ * _ = (:@)
  _ - _ = (:@)

  abs _ = (:@)
  negate _ = (:@)
  signum _ = (:@)
  
  fromInteger _ = (:@)

instance (Num a, Num (TList as)) => Num (TList (a ': as)) where
  (a :+ as) + (b :+ bs) =
    (a + b) :+ (as + bs)

  (a :+ as) - (b :+ bs) =
    (a - b) :+ (as - bs)

  (a :+ as) * (b :+ bs) =
    (a * b) :+ (as * bs)

  abs (a :+ as) = (abs a) :+ (abs as)
  negate (a :+ as) = (negate a) :+ (negate as)
  signum (a :+ as) = (signum a) :+ (signum as)

  fromInteger a = fromInteger a :+ fromInteger a

instance Show (TList '[]) where
  show _ = ""

instance (Show (TList as), Show a) => Show (TList (a ': as)) where
  show (a :+ as) = show a ++ "\n" ++ show as


