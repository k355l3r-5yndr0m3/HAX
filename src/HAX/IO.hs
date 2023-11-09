{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.IO where

import Codec.Picture

import Data.Proxy
import Data.Vector.Storable

import Foreign

import HAX.Tensor
import HAX.PjRt (defaultDevice)


import GHC.TypeLits 
import GHC.IO.Unsafe


class Pixel t => ImageLoader t where
  type ImageTensor (h :: Nat) (w :: Nat) t = r | r -> w h t

  tensorFromImage :: (KnownNat h, KnownNat w) => Image t -> ImageTensor h w t
  imageFromTensor :: (KnownNat h, KnownNat w) => ImageTensor h w t -> Image t


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

