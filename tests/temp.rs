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

        // On prépare nos paramètres 
        let args = main::Options {
            image: "assets/kit.jpeg".to_string(),
            output: output.to_string(),
            tiles: "assets/images".to_string(),
            scaling: 1,
            tile_size: 25,
            remove_used: false,
            verbose: false,
            simd: true,     // on active le simd
            num_thread: 1,
        };

        main::compute_mosaic(args);

        assert!(Path::new(output).exists(),"compute_mosaic() avec x86 n'a pas généré de out ");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
        let output = "out_test_arm.png";

        let _ = std::fs::remove_file(output);

        let args = main::Options {
            image: "assets/kit.jpeg".to_string(),
            output: output.to_string(),
            tiles: "assets/images".to_string(),
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
        let output = "out_test_generic.png";


        let args = main::Options {
            image: "assets/kit.jpeg".to_string(),
            output: output.to_string(),
            tiles: "assets/images".to_string(),
            scaling: 1,
            tile_size: 25,
            remove_used: false,
            verbose: false,
            simd: false,  // pas de simd
            num_thread: 1,
        };

        main::compute_mosaic(args);

        //  On charge l'image générée grace a ma fonction en entete
        let mosaic_généré = load_rgb(output);

        let ground_trut_kit = load_rgb("assets/ground-truth-kit.png");

        assert_eq!(mosaic_généré.width(),ground_trut_kit.width(),"largeur pas bonnnnnnn");

        assert_eq!(mosaic_généré.height(),ground_trut_kit.height(),"hauteur pas bon");

        // pour etre sur , on va comparé pixel par pixel en parcourant les 2 images
        for (pix_mos, pix_gt) in mosaic_généré.pixels().zip(ground_trut_kit.pixels()) {
            assert_eq!(pix_mos.0,pix_gt.0,"Probéme : aumoins un pixel ne correspond pas");}
    }
}
