{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
module HAX.HList where
import Data.Kind
import Data.Tuple

data HList (t :: [Type]) where
  (:@) :: HList '[]
  (:+) :: a -> HList as -> HList (a ': as)
infixr 5 :+, :@

class HListTuple l t | l -> t, t -> l where
  tlist2tup :: l -> t
  tup2tlist :: t -> l

instance HListTuple (HList '[]) () where
  tlist2tup _ = ()
  tup2tlist _ = (:@)

instance HListTuple (HList '[a]) (Solo a) where
  tlist2tup (a :+ (:@)) = MkSolo a
  tup2tlist (MkSolo a)  = a :+ (:@)

instance HListTuple (HList '[a, b]) (a, b) where
  tlist2tup (a :+ b :+ (:@)) = (a, b)
  tup2tlist (a, b) = a :+ b :+ (:@)

instance HListTuple (HList '[a, b, c]) (a, b, c) where
  tlist2tup (a :+ b :+ c :+ (:@)) = (a, b, c)
  tup2tlist (a, b, c) = a :+ b :+ c :+ (:@)

class HListConcat a b where
  type Concat a b :: [Type]
  (++:) :: HList a -> HList b -> HList (Concat a b)

instance HListConcat '[] b where
  type Concat '[] b = b
  _ ++: b = b

instance HListConcat as b => HListConcat (a ': as) b where
  type Concat (a ': as) b = a ': Concat as b
  (a :+ as) ++: b = a :+ (as ++: b)

instance Num (HList '[]) where
  _ + _ = (:@)
  _ * _ = (:@)
  _ - _ = (:@)

  abs _ = (:@)
  negate _ = (:@)
  signum _ = (:@)
  
  fromInteger _ = (:@)

instance (Num a, Num (HList as)) => Num (HList (a ': as)) where
  (a :+ as) + (b :+ bs) =
    (a + b) :+ (as + bs)

  (a :+ as) - (b :+ bs) =
    (a - b) :+ (as - bs)

  (a :+ as) * (b :+ bs) =
    (a * b) :+ (as * bs)

  abs (a :+ as) = abs a :+ abs as
  negate (a :+ as) = negate a :+ negate as
  signum (a :+ as) = signum a :+ signum as

  fromInteger a = fromInteger a :+ fromInteger a

instance Show (HList '[]) where
  show _ = ""

instance (Show (HList as), Show a) => Show (HList (a ': as)) where
  show (a :+ as) = show a ++ "\n" ++ show as

class HListLen l where 
  hlistLen  :: HList l -> Int

instance HListLen '[] where
  hlistLen _ = 0

instance HListLen as => HListLen (a ': as) where
  hlistLen (_ :+ as) = 1 + hlistLen as

