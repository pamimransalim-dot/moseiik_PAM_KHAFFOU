/* #[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {
        // TODO
        // test avx2 or sse2 if available
        assert!(false);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
        //TODO
        assert!(false);
    }

    #[test]
    fn test_generic() {
        //TODO
        assert!(false);
    }
}
*/

use moseiik::main;            
use image::{RgbImage, ImageReader};
use std::path::Path;

// Fonction  charger une image rgb
fn load_rgb(path: &str) -> RgbImage {
    ImageReader::open(path)
        .expect(&format!("Impossible d'ouvrir {}", path))
        .decode()
        .expect(&format!("Impossible de décoder {}", path))
        .into_rgb8()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {
        let output = "out_test_x86.png";

        // On prépare les paramètres comme si on était en ligne de commande
        let args = main::Options {
            image: "assets/kit.jpeg".to_string(),
            output: output.to_string(),
            tiles: "assets/tiles-small".to_string(),
            scaling: 1,
            tile_size: 5,
            remove_used: false,
            verbose: false,
            simd: true,     // on active le simd
            num_thread: 1,
        };

        main::compute_mosaic(args);

        assert!(
            Path::new(output).exists(),"compute_mosaic() avec x86 n'a pas généré de out ");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
        let output = "out_test_arm.png";

        let _ = std::fs::remove_file(output);

        let args = main::Options {
            image: "assets/kit.jpeg".to_string(),
            output: output.to_string(),
            tiles: "assets/tiles-small".to_string(),
            scaling: 1,
            tile_size: 5,
            remove_used: false,
            verbose: false,
            simd: true,     // simd neon
            num_thread: 1,
        };

        main::compute_mosaic(args);

        assert!(Path::new(output).exists(),"compute_mosaic() avec ARM n'a pas généré de out");
    }

    
    #[test]
    fn test_generic() {
        let output = "out_test_ground_truth.png";


        let args = main::Options {
            image: "assets/kit.jpeg".to_string(),
            output: output.to_string(),
            tiles: "assets/tiles-small".to_string(),
            scaling: 1,
            tile_size: 5,
            remove_used: false,
            verbose: false,
            simd: false,  // pas de simd
            num_thread: 1,
        };

        main::compute_mosaic(args);

        //  On charge l'image générée grace a ma fonction en entete
        let mosaic = load_rgb(output);

        let ground_truth = load_rgb("assets/ground-truth-kit.png");

        assert_eq!(mosaic.width(),ground_truth.width(),"largeur inégale avec la mosaïque générée");

        assert_eq!(mosaic.height(),ground_truth.height(),"hauteur inégale avec la mosaïque générée");

        // pour etre sur , on va comparé pixel par pixel
        for (pix_mos, pix_gt) in mosaic.pixels().zip(ground_truth.pixels()) {
            assert_eq!(pix_mos.0,pix_gt.0,"Les images diffèrent : aumoins un pixel ne correspond pas");}
    }
}
