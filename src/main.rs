use clap::Parser;
use image::{
    imageops::{resize, FilterType::Nearest},
    GenericImage, GenericImageView, ImageReader, RgbImage,
};
use std::time::Instant;
use std::{
    error::Error,
    fs,
    ops::Deref,
    sync::{Arc, Mutex},
};
use threadpool::ThreadPool;
use threadpool_scope::scope_with;

#[derive(Debug, Parser)]

/// structure pour stocker largeur/hauteur des tiles 
struct Size {
    width: u32,
    height: u32,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Options {
    /// Location of the target image
    #[arg(short, long)]
    pub image: String,

    /// Saved result location
    #[arg(short, long, default_value_t=String::from("out.png"))]
    pub output: String,

    /// Location of the tiles
    #[arg(short, long)]
    pub tiles: String,

    /// Scaling factor of the image
    #[arg(long, default_value_t = 1)]
    pub scaling: u32,

    /// Size of the tiles
    #[arg(long, default_value_t = 5)]
    pub tile_size: u32,

    /// Remove used tile
    #[arg(short, long)]
    pub remove_used: bool,

    #[arg(short, long)]
    pub verbose: bool,

    /// Use SIMD when available
    #[arg(short, long)]
    pub simd: bool,

    /// Specify number of threads to use, leave blank for default
    #[arg(short, long, default_value_t = 1)]
    pub num_thread: usize,
}

/// Compte le nombre d'images disponibles dans le dossier des tiles
fn count_available_tiles(images_folder: &str) -> i32 {
    match fs::read_dir(images_folder) {
        Ok(t) => return t.count() as i32,
        Err(_) => return -1,
    };
}

/// Charge toutes les tiles du dossier, les redimensionne à la bonne taille et renvoie un `Vec<RgbImage>`.
fn prepare_tiles(
    images_folder: &str,
    tile_size: &Size,
    verbose: bool,
) -> Result<Vec<RgbImage>, Box<dyn Error>> {
    let nb_tiles: usize = count_available_tiles(images_folder) as usize;

    let mut tile_names: Vec<_> = fs::read_dir(images_folder)
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    tile_names.sort_by_key(|dir| dir.path());

    // Declare a vector in a nb_tiles wide memory segment
    let tiles = Arc::new(Mutex::new(Vec::with_capacity(nb_tiles) as Vec<RgbImage>));
    // Actually allocate memory
    tiles.lock().unwrap().resize(nb_tiles, RgbImage::new(0, 0));

    let now = Instant::now();
    let pool = ThreadPool::new(num_cpus::get());
    let tile_width = tile_size.width;
    let tile_height = tile_size.height;

    // for image_path in image_paths
    for (index, image_path) in (0..=nb_tiles - 1).zip(tile_names) {
        let tiles = Arc::clone(&tiles);
        pool.execute(move || {
            let tile_result = || -> Result<RgbImage, Box<dyn Error>> {
                Ok(ImageReader::open(image_path.path())?.decode()?.into_rgb8())
            };

            let tile = match tile_result() {
                Ok(t) => t,
                Err(_) => return,
            };

            let tile = resize(&tile, tile_width, tile_height, Nearest);
            tiles.lock().unwrap()[index] = tile;
        });
    }
    // On attend la fin de tous les threads du pool
    pool.join();

    println!(
        "\n{} elements in {} seconds",
        tiles.lock().unwrap().len(),
        now.elapsed().as_millis() as f32 / 1000.0
    );

    if verbose {
        println!("");
    }
    let res = tiles.lock().unwrap().deref().to_owned();
    return Ok(res);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "avx")]
unsafe fn l1_x86_avx2(im1: &RgbImage, im2: &RgbImage) -> i32 {
    // Suboptimal performance due to the use of _mm256_loadu_si256.
    use std::arch::x86_64::{
        __m256i,
        _mm256_extract_epi16, //AVX2
        _mm256_load_si256,    //AVX
        _mm256_loadu_si256,   //AVX
        _mm256_sad_epu8,      //AVX2
    };

    let stride = std::mem::size_of::<__m256i>();

    let tile_size = (im1.width() * im1.height()) as usize;
    let nb_sub_pixel = tile_size * 3;

    let im1 = im1.as_raw();
    let im2 = im2.as_raw();

    let mut result: i32 = 0;

    for i in (0..nb_sub_pixel - stride).step_by(stride) {
        // Get pointer to data
        let p_im1: *const __m256i =
            std::mem::transmute::<*const u8, *const __m256i>(std::ptr::addr_of!(im1[i as usize]));
        let p_im2: *const __m256i =
            std::mem::transmute::<*const u8, *const __m256i>(std::ptr::addr_of!(im2[i as usize]));

        // Load data to ymm
        let ymm_p1 = _mm256_loadu_si256(p_im1);
        let ymm_p2 = _mm256_load_si256(p_im2);

        // Do abs(a-b) and horizontal add, results are stored in lower 16 bits of each 64 bits groups
        let ymm_sub_abs = _mm256_sad_epu8(ymm_p1, ymm_p2);

        let res_0 = _mm256_extract_epi16(ymm_sub_abs, 0);
        let res_1 = _mm256_extract_epi16(ymm_sub_abs, 4);
        let res_2 = _mm256_extract_epi16(ymm_sub_abs, 8);
        let res_3 = _mm256_extract_epi16(ymm_sub_abs, 12);

        result += res_0 + res_1 + res_2 + res_3;
    }

    // now do the remainder manually
    let remainder = nb_sub_pixel % stride;
    for i in nb_sub_pixel - remainder..nb_sub_pixel {
        let p1: u8 = im1[i as usize];
        let p2: u8 = im2[i as usize];

        result += i32::abs((p1 as i32) - (p2 as i32));
    }

    return result;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn l1_x86_sse2(im1: &RgbImage, im2: &RgbImage) -> i32 {
    // Only works if data is 16 bytes-aligned, which should be the case.
    // In case of crash due to unaligned data, swap _mm_load_si128 for _mm_loadu_si128.
    use std::arch::x86_64::{
        __m128i,
        _mm_extract_epi16, //SSE2
        _mm_load_si128,    //SSE2
        _mm_sad_epu8,      //SSE2
    };

    let stride = std::mem::size_of::<__m128i>();

    let tile_size = (im1.width() * im1.height()) as usize;
    let nb_sub_pixel = tile_size * 3;

    let im1 = im1.as_raw();
    let im2 = im2.as_raw();

    let mut result: i32 = 0;

    for i in (0..nb_sub_pixel - stride).step_by(stride) {
        // Get pointer to data
        let p_im1: *const __m128i =
            std::mem::transmute::<*const u8, *const __m128i>(std::ptr::addr_of!(im1[i as usize]));
        let p_im2: *const __m128i =
            std::mem::transmute::<*const u8, *const __m128i>(std::ptr::addr_of!(im2[i as usize]));

        // Load data to xmm
        let xmm_p1 = _mm_load_si128(p_im1);
        let xmm_p2 = _mm_load_si128(p_im2);

        // Do abs(a-b) and horizontal add, results are stored in lower 16 bits of each 64 bits groups
        let xmm_sub_abs = _mm_sad_epu8(xmm_p1, xmm_p2);

        let res_0 = _mm_extract_epi16(xmm_sub_abs, 0);
        let res_1 = _mm_extract_epi16(xmm_sub_abs, 4);

        result += res_0 + res_1;
    }

    // now do the remainder manually
    let remainder = nb_sub_pixel % stride;
    for i in nb_sub_pixel - remainder..nb_sub_pixel {
        let p1: u8 = im1[i];
        let p2: u8 = im2[i];

        result += i32::abs((p1 as i32) - (p2 as i32));
    }

    return result;
}

fn l1_generic(im1: &RgbImage, im2: &RgbImage) -> i32 {
    im1.iter()
        .zip(im2.iter())
        .fold(0, |res, (a, b)| res + i32::abs((*a as i32) - (*b as i32)))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_neon(im1: &RgbImage, im2: &RgbImage) -> i32 {
    use std::arch::aarch64::uint8x16_t;
    use std::arch::aarch64::vabdq_u8; // Absolute subtract
    use std::arch::aarch64::vaddlvq_u8; // horizontal add
    use std::arch::aarch64::vld1q_u8; // Load instruction

    let stride = std::mem::size_of::<uint8x16_t>();

    let tile_size = (im1.width() * im1.height()) as usize;
    let nb_sub_pixel = tile_size * 3;

    let im1 = im1.as_raw();
    let im2 = im2.as_raw();

    let mut result: i32 = 0;

    for i in (0..nb_sub_pixel - stride).step_by(stride as usize) {
        // get pointer to data
        let p_im1: *const u8 = std::ptr::addr_of!(im1[i as usize]);
        let p_im2: *const u8 = std::ptr::addr_of!(im2[i as usize]);

        // load data to xmm
        let xmm1: uint8x16_t = vld1q_u8(p_im1);
        let xmm2: uint8x16_t = vld1q_u8(p_im2);

        // get absolute difference
        let xmm_abs_diff: uint8x16_t = vabdq_u8(xmm1, xmm2);

        // reduce with horizontal add
        result += vaddlvq_u8(xmm_abs_diff) as i32;
    }

    // now do the remainder manually
    let remainder = nb_sub_pixel % stride;
    for i in nb_sub_pixel - remainder..nb_sub_pixel {
        let p1: u8 = im1[i as usize];
        let p2: u8 = im2[i as usize];

        result += i32::abs((p1 as i32) - (p2 as i32));
    }

    return result;
}

fn l1(im1: &RgbImage, im2: &RgbImage, simd_flag: bool, verbose: bool) -> i32 {
    return unsafe { get_optimal_l1(simd_flag, verbose)(im1, im2) };
}

unsafe fn get_optimal_l1(simd_flag: bool, verbose: bool) -> unsafe fn(&RgbImage, &RgbImage) -> i32 {
    static mut FN_POINTER: unsafe fn(&RgbImage, &RgbImage) -> i32 = l1_generic;

    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        if simd_flag {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    if verbose {
                        println!("{}[2K\rUsing AVX2 SIMD.", 27 as char);
                    }
                    FN_POINTER = l1_x86_avx2;
                } else if is_x86_feature_detected!("sse2") {
                    if verbose {
                        println!("{}[2K\rUsing SSE2 SIMD.", 27 as char);
                    }
                    FN_POINTER = l1_x86_sse2;
                } else {
                    if verbose {
                        println!("{}[2K\rNot using SIMD.", 27 as char);
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::is_aarch64_feature_detected;
                if is_aarch64_feature_detected!("neon") {
                    if verbose {
                        println!("{}[2K\rUsing NEON SIMD.", 27 as char);
                    }
                    FN_POINTER = l1_neon;
                }
            }
        }
    });

    return FN_POINTER;
}

fn prepare_target(
    image_path: &str,
    scale: u32,
    tile_size: &Size,
) -> Result<RgbImage, Box<dyn Error>> {
    let target = ImageReader::open(image_path)?.decode()?.into_rgb8();
    let target = resize(
        &target,
        target.width() * scale,
        target.height() * scale,
        Nearest,
    );
    Ok(target
        .view(
            0,
            0,
            target.width() - target.width() % tile_size.width,
            target.height() - target.height() % tile_size.height,
        )
        .to_image())
}

pub fn compute_mosaic(args: Options) {
    let tile_size = Size {
        width: args.tile_size,
        height: args.tile_size,
    };

    let (target_size, target) = match prepare_target(&args.image, args.scaling, &tile_size) {
        Ok(t) => (
            Size {
                width: t.width(),
                height: t.height(),
            },
            Arc::new(Mutex::new(t)),
        ),
        Err(e) => panic!("Error opening {}. {}", args.image, e),
    };

    let nb_available_tiles = count_available_tiles(&args.tiles);
    let nb_required_tiles: i32 =
        ((target_size.width / tile_size.width) * (target_size.height / tile_size.height)) as i32;
    if args.remove_used && nb_required_tiles > nb_available_tiles {
        panic!(
            "{} tiles required, found {}.",
            nb_required_tiles, nb_available_tiles
        )
    }

    let tiles = &prepare_tiles(&args.tiles, &tile_size, args.verbose).unwrap();
    if args.verbose {
        println!("w: {}, h: {}", target_size.width, target_size.height);
    }

    let now = Instant::now();
    let pool = ThreadPool::new(args.num_thread);
    scope_with(&pool, |scope| {
        for w in 0..target_size.width / tile_size.width {
            let target = Arc::clone(&target);
            scope.execute(move || {
                for h in 0..target_size.height / tile_size.height {
                    if args.verbose {
                        print!(
                            "\rBuilding image: {} / {} : {} / {}",
                            w,
                            target_size.width / tile_size.width,
                            h,
                            target_size.height / tile_size.height
                        );
                    }
                    let mut best_tile = 0;
                    let mut min_error = i32::MAX;
                    let target_tile = &(target
                        .lock()
                        .unwrap()
                        .view(
                            tile_size.width * w,
                            tile_size.height * h,
                            tile_size.width,
                            tile_size.height,
                        )
                        .to_image());

                    for (i, tile) in tiles.iter().enumerate() {
                        let error = l1(tile, &target_tile, args.simd, args.verbose);

                        if error < min_error {
                            min_error = error;
                            best_tile = i;
                        }
                    }

                    target
                        .lock()
                        .unwrap()
                        .copy_from(&tiles[best_tile], w * tile_size.width, h * tile_size.height)
                        .unwrap();
                }
            });
        }
    });
    println!("\n{} seconds", now.elapsed().as_millis() as f32 / 1000.0);

    target.lock().unwrap().save(args.output).unwrap();
}

fn main() {
    let args = Options::parse();
    compute_mosaic(args);
}
/* 

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn unit_test_x86() {
        // TODO
        assert!(true);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn unit_test_aarch64() {
        assert!(true);
    }

    #[test]
    fn unit_test_generic() {
        // TODO
        assert!(true);
    }
}
*/

#[cfg(test)]
mod tests {
   
    use super::*;
    use image::ImageReader;

    fn create_simple_image(width: u32, height: u32, value: u8) -> RgbImage {
        let mut img = RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            // ici je mets rgb à la même valeur pour faciliter mon calcul à la main 
            *pixel = image::Rgb([value, value, value]);
        }
        img
    }

    #[test]
    fn test_l1_generic_valeur_connue() {
        // ici je crée 2 images de même taille et à valeur de pixel constant
        let im1 = create_simple_image(2, 2, 0);
        let im2 = create_simple_image(2, 2, 10);
        // donc distance L1 = 4 * 3 * |10 - 0| = 120
        let distance = l1_generic(&im1, &im2);
        assert_eq!(distance, 120);
    }

    /// ici on teste la fonction avec la meilleure implémentation (SIMD ou générique)
    /// mais le résultat doit être le même que `l1_generic`
    
    #[test]
    fn test_l1_wrapper_egal_a_l1_generic() {
        let im1 = create_simple_image(4, 4, 5);
        let im2 = create_simple_image(4, 4, 20);

        let attendu = l1_generic(&im1, &im2);

        // alternative 1 : on n'active PAS le SIMD
        let d_sans_simd = l1(&im1, &im2, false, false);
        assert_eq!(d_sans_simd, attendu);

        // alternative 2 : on active le SIMD 
        // si simd n'est pas disponible, l1 retombera sur l'implémentation générique donc dans TOUS les cas le résultat doit être le même
        let d_avec_simd = l1(&im1, &im2, true, false);
        assert_eq!(d_avec_simd, attendu);
    }

   

// a l'execution j'ai eu cette echec : 
//failures:
//---- main::tests::test_l1_simd_x86_egal_generic stdout ----
//thread 'main::tests::test_l1_simd_x86_egal_generic' panicked at src\main.rs:531:17:
//assertion `left == right` failed
//left: 4800
//right: 5760
//note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
// solution : modifications de for i in (0..nb_sub_pixel - stride).step_by(stride) en for i in (0..=nb_sub_pixel - stride).step_by(stride) dans les fonction l1_x86_avx2 ,l1_x86_sse2 qui permet de traiter le dernier bloc de 32octets oublié

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_l1_simd_x86_egal_generic() {
        // Cette fonction est fournie par la lib standard : elle permet de savoir si le processeur supporte AVX2 / SSE2.
        use std::arch::is_x86_feature_detected;

        let im1 = create_simple_image(5, 5, 0);
        let im2 = create_simple_image(5, 5, 30);

        let attendu = l1_generic(&im1, &im2);

        unsafe {
            // on teste l1_x86_avx2
            if is_x86_feature_detected!("avx2") {
                let d_avx2 = l1_x86_avx2(&im1, &im2);
                assert_eq!(d_avx2, attendu);
            }

            // on teste l1_x86_sse2
            if is_x86_feature_detected!("sse2") {
                let d_sse2 = l1_x86_sse2(&im1, &im2);
                assert_eq!(d_sse2, attendu);
            }
        }
    }

    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_l1_neon_egal_generic() {
        use std::arch::is_aarch64_feature_detected;

        let im1 = create_simple_image(8, 8, 0);
        let im2 = create_simple_image(8, 8, 30);

        let attendu = l1_generic(&im1, &im2);

        unsafe {
            if is_aarch64_feature_detected!("neon") {
                let d_neon = l1_neon(&im1, &im2);
                assert_eq!(d_neon, attendu);
            }
        }
    }

   

   
    /// Idée du test : on appelle directement prepare_tiles ,ensuite on  vérifie qu'on a bien au moins UNE tile, et aprés on vérifie que toutes les tiles ont la bonne taille (tile_size)
    ///  et enfin on vérifie que  nombre de tiles = nombre de fichiers
    #[test]
    fn test_prepare_tiles_sur_assets_tiles_small() {

        let tiles_folder = "assets/tiles-small";
        let tile_size = Size {
            width: 5,
            height: 5,
        };

        // on call la fonction à tester ; Si echec un message clair
        let tiles = prepare_tiles(tiles_folder, &tile_size, false)
            .expect("prepare_tiles a lamentablement loseeeeeeeeeeee (dossier assets/tiles-small manquant ?)");

        assert!(!tiles.is_empty(),"Aucune tile chargéeeeeee "
        );

        // Chaque tile a la bonne taille 
        for tile in &tiles {
            assert_eq!(tile.width(), tile_size.width);
            assert_eq!(tile.height(), tile_size.height);
        }

        //nombre d'éléments renvoyés = nombre de fichiers trouvés dans le dossier?
        let nb_compte_par_fonction = count_available_tiles(tiles_folder);
        assert_eq!(tiles.len() as i32,nb_compte_par_fonction,"nombre d'éléments renvoyés != nombre de fichiers trouvés dans le dossier");
    }

    
    /// Idée du test  :on lit l'image originale pour connaître sa taille ,ensuite calcule à la main la taille attendue après scaling de 2 + rognage et enfin on vérifie que `prepare_target` renvoie bien cette taille
    #[test]
    fn test_prepare_target_scale_2() {
        let image_path = "assets/target-small.png";

        // On lit l'image d'origine 
        let original = ImageReader::open(image_path)
            .expect("Impossible d'ouvrir assets/target-small.png")
            .decode()
            .unwrap()
            .into_rgb8();

        let scale = 2;
        let tile_size = Size {
            width: 5,
            height: 5,
        };

        // Taille after scaling sans rogner 
        let scaled_w = original.width() * scale;
        let scaled_h = original.height() * scale;

        // taille apreés rogne (qui doit etre un multile de tile_size)
        let expected_w = scaled_w - (scaled_w % tile_size.width);
        let expected_h = scaled_h - (scaled_h % tile_size.height);

        let prepared = prepare_target(image_path, scale, &tile_size).expect("prepare_target a échoué");

        //  dimensions = dimensions calculé
        assert_eq!(prepared.width(), expected_w);
        assert_eq!(prepared.height(), expected_h);

        // dimensions = dimensions multiple de tile_size
        assert_eq!(prepared.width() % tile_size.width, 0);
        assert_eq!(prepared.height() % tile_size.height, 0);
    }


    /// test sur prepare_target pour scale = 1
    #[test]
    fn test_prepare_target_scale_1_multiple_de_tile() {
        let image_path = "assets/target-small.png";

        let tile_size = Size {
            width:5,
            height:5,
        };

        let prepared =
            prepare_target(image_path, 1, &tile_size).expect("prepare_target a échoué");

        // l'image n'est pas vide !
        assert!(prepared.width() > 0);
        assert!(prepared.height() > 0);

        // taille multiple de 5
        assert_eq!(prepared.width() % tile_size.width, 0);
        assert_eq!(prepared.height() % tile_size.height, 0);
    }
}
