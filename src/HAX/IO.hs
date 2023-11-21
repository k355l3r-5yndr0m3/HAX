{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.IO where

import Codec.Picture

import Data.Bool (bool)
import Data.Proxy
import Data.Vector.Storable hiding (product, (++))
import Data.Maybe (fromMaybe)

import Foreign

import HAX.Tensor
import HAX.PjRt (client, defaultDevice, clientBufferFromHostBufferGC)
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.BufferType

import GHC.TypeLits 
import GHC.IO.Unsafe

import System.IO
import System.Endian

class Pixel t => ImageLoader t where
  type ImageTensor (h :: Nat) (w :: Nat) t = r | r -> w h t

  tensorFromImage :: (KnownNat h, KnownNat w) => Image t -> ImageTensor h w t
  imageFromTensor :: (KnownNat h, KnownNat w) => ImageTensor h w t -> Image t

data IDXDataType = IDXUnsignedByte | IDXSignedByte | IDXShort | IDXInt | IDXFloat | IDXDouble deriving Enum

-- TODO: Implement more failure condition
readIDXFile :: FilePath -> IO (Maybe Tensor')
readIDXFile filepath = withBinaryFile filepath ReadMode (\handle -> do  
  (rank, datatype) <- allocaArray 4 (\magic -> do 
    _ <- hGetBuf handle (magic :: Ptr Word8) 4
    t <- peekElemOff magic 2
    r <- fromIntegral <$> peekElemOff magic 3
    return (r, case t of 0x08 -> Just IDXUnsignedByte
                         0x09 -> Just IDXSignedByte
                         0x0B -> Just IDXShort
                         0x0C -> Just IDXInt
                         0x0D -> Just IDXFloat
                         0x0E -> Just IDXDouble
                         _    -> Nothing))
  shape <- allocaArray rank (\buf -> do 
    _ <- hGetBuf handle buf (4 * rank)
    fmap (fromIntegral . fromBE32) <$> peekArray rank buf)
  bool (
    case datatype of 
      Nothing -> return Nothing 
      Just dt -> Just . Tensor' <$> do 
        let elemcount = product shape
        buffer :: Ptr () <- case dt of 
          IDXUnsignedByte -> readToBuffer handle elemcount (id :: Word8  -> Word8)
          IDXSignedByte   -> readToBuffer handle elemcount (id :: Int8   -> Int8)
          IDXShort        -> readToBuffer handle elemcount fromBE16
          IDXInt          -> readToBuffer handle elemcount fromBE32
          IDXFloat        -> readToBuffer handle elemcount (id :: Float  -> Float)
          IDXDouble       -> readToBuffer handle elemcount (id :: Double -> Double)
        clientBufferFromHostBufferGC client buffer (case dt of 
          IDXUnsignedByte -> u8
          IDXSignedByte   -> s8
          IDXShort        -> s16
          IDXInt          -> s32
          IDXFloat        -> f32
          IDXDouble       -> f64) (Shape (fromIntegral <$> shape)) defaultDevice) (return Nothing) =<< hIsEOF handle
  )
  where readToBuffer :: forall a. Storable a => Handle -> Int -> (a -> a) -> IO (Ptr ())
        readToBuffer handle elemcount fromBE = do 
          buffer <- mallocArray elemcount
          _ <- hGetBuf handle buffer (elemcount * sizeOf (undefined :: a))
          sequence_ [pokeElemOff buffer i . fromBE =<< peekElemOff buffer i | i <- [0..elemcount - 1]]
          return $ castPtr buffer

readIDXFile' :: FilePath -> IO Tensor' 
readIDXFile' filepath = fromMaybe (error $ filepath ++ " is not an idx file or an invalid idx file") <$> readIDXFile filepath

--Dataset 
--  features:
--    shuffling 
--    preprocessing
--    train/validation set
--    batching
--
--
--A batch
--  n samples
--  similar to folding
--  a singular iteration through the dataset is f :: a -> batch -> a
--  foldl f initial_state dataset
--
--collating
--  from dynamic images/text/etc to static type tensor
--  [sample] -> (state -> batch -> state)
--
--train    :: state -> batch -> state
--validate :: state -> batch -> evaluation
--
--collator :: [sample] -> batch
--
--preprocessing :> collator ?
--agumentation :> collator ? 
--

class Dataset s where
  type Sample s

  loadDataset  :: FilePath -> IO (Either String s)
  loadDataset' :: FilePath -> IO s

  datasetSize   :: s -> Int
  sampleDataset :: s -> Int -> Sample s




data Dataloader s 








































instance ImageLoader Pixel8 where
  type ImageTensor h w Pixel8 = Tensor [h, w] Word8

  tensorFromImage :: forall h w. (KnownNat h, KnownNat w) => Image Pixel8 -> Tensor [h, w] Word8
  tensorFromImage image = unsafePerformIO $ do
    buffer <- mallocArray (channel * width * height)
    sequence_ [do 
      pokeElemOff buffer (0 + channel * (x + width * y)) $ pixelAt image x y| x <- [0..min width (imageWidth image) - 1], y <- [0..min height (imageHeight image) - 1]]
    tensorFromHostBufferGC defaultDevice buffer
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h) 
          channel = 1

  imageFromTensor :: forall h w. (KnownNat h, KnownNat w) => Tensor [h, w] Word8 -> Image Pixel8
  imageFromTensor image = unsafePerformIO $ do 
    buffer <- newForeignPtr finalizerFree . snd =<< tensorToHostBuffer image
    return $ Image { imageWidth = width, imageHeight = height, imageData = unsafeFromForeignPtr0 buffer (width * height * channel) }
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h)
          channel = 1
  

instance ImageLoader PixelRGB8 where
  type ImageTensor h w PixelRGB8 = Tensor [h, w, 3] Word8

  tensorFromImage :: forall h w. (KnownNat h, KnownNat w) => Image PixelRGB8 -> Tensor [h, w, 3] Word8
  tensorFromImage image = unsafePerformIO $ do
    buffer <- mallocArray (channel * width * height)
    sequence_ [do 
      let PixelRGB8 red green blue = pixelAt image x y
      pokeElemOff buffer (0 + channel * (x + width * y)) red
      pokeElemOff buffer (1 + channel * (x + width * y)) green
      pokeElemOff buffer (2 + channel * (x + width * y)) blue| x <- [0..min width (imageWidth image) - 1], y <- [0..min height (imageHeight image) - 1]]
    tensorFromHostBufferGC defaultDevice buffer
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h) 
          channel = 3

  imageFromTensor :: forall h w. (KnownNat h, KnownNat w) => Tensor [h, w, 3] Word8 -> Image PixelRGB8
  imageFromTensor image = unsafePerformIO $ do 
    buffer <- newForeignPtr finalizerFree . snd =<< tensorToHostBuffer image
    return $ Image { imageWidth = width, imageHeight = height, imageData = unsafeFromForeignPtr0 buffer (width * height * channel) }
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h)
          channel = 3

instance ImageLoader PixelF where
  type ImageTensor h w PixelF = Tensor [h, w] Float

  tensorFromImage :: forall h w. (KnownNat h, KnownNat w) => Image PixelF -> Tensor [h, w] Float
  tensorFromImage image = unsafePerformIO $ do
    buffer <- mallocArray (channel * width * height)
    sequence_ [pokeElemOff buffer (0 + channel * (x + width * y)) $ pixelAt image x y | x <- [0..min width (imageWidth image) - 1], y <- [0..min height (imageHeight image) - 1]]
    tensorFromHostBufferGC defaultDevice buffer
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h) 
          channel = 1

  imageFromTensor :: forall h w. (KnownNat h, KnownNat w) => Tensor [h, w] Float -> Image PixelF
  imageFromTensor image = unsafePerformIO $ do 
    buffer <- newForeignPtr finalizerFree . snd =<< tensorToHostBuffer image
    return $ Image { imageWidth = width, imageHeight = height, imageData = unsafeFromForeignPtr0 buffer (width * height * channel) }
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h)
          channel = 1

instance ImageLoader PixelRGBF where
  type ImageTensor h w PixelRGBF = Tensor [h, w, 3] Float

  tensorFromImage :: forall h w. (KnownNat h, KnownNat w) => Image PixelRGBF -> Tensor [h, w, 3] Float
  tensorFromImage image = unsafePerformIO $ do
    buffer <- mallocArray (channel * width * height)
    sequence_ [do 
      let PixelRGBF red green blue = pixelAt image x y
      pokeElemOff buffer (0 + channel * (x + width * y)) red
      pokeElemOff buffer (1 + channel * (x + width * y)) green
      pokeElemOff buffer (2 + channel * (x + width * y)) blue| x <- [0..min width (imageWidth image) - 1], y <- [0..min height (imageHeight image) - 1]]
    tensorFromHostBufferGC defaultDevice buffer
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h) 
          channel = 3

  imageFromTensor :: forall h w. (KnownNat h, KnownNat w) => Tensor [h, w, 3] Float -> Image PixelRGBF
  imageFromTensor image = unsafePerformIO $ do 
    buffer <- newForeignPtr finalizerFree . snd =<< tensorToHostBuffer image
    return $ Image { imageWidth = width, imageHeight = height, imageData = unsafeFromForeignPtr0 buffer (width * height * channel) }
    where width   = fromInteger $ natVal (Proxy :: Proxy w)
          height  = fromInteger $ natVal (Proxy :: Proxy h)
          channel = 3

