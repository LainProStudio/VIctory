use super::*;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/// A one-thread compute shader containing only s_endpgm, with a private
/// metadata byte outside the DXIL part to distinguish container variants.
fn container(metadata: u8) -> Vec<u8> {
    let mut payload = vec![0u8; 0x68 + 0x54 + 0x34]; // stage slots, CS record, CS metadata
    payload[..8].copy_from_slice(&6u64.to_le_bytes());
    for offset in [8, 16, 24, 32, 40, 48, 72] {
        payload[offset..offset + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    }
    let cs = 0x68;
    let code_offset = payload.len() as u64;
    payload[0x60..cs].copy_from_slice(&(cs as u64).to_le_bytes());
    payload[cs..cs + 8].copy_from_slice(&code_offset.to_le_bytes());
    payload[cs + 8..cs + 16].copy_from_slice(&4u64.to_le_bytes());
    for axis in payload[cs + 0x54 + 0x28..].chunks_exact_mut(4) {
        axis.copy_from_slice(&1u32.to_le_bytes());
    }
    payload.extend_from_slice(&0xBF81_0000u32.to_le_bytes()); // s_endpgm

    let mut fssm = b"FSSM".to_vec();
    fssm.extend(0x101u32.to_le_bytes());
    fssm.extend((payload.len() as u64).to_le_bytes());
    fssm.extend(payload);

    // One uncompressed DEFLATE block in a zlib stream, with an Adler-32 trailer.
    let len = u16::try_from(fssm.len()).unwrap();
    let mut zlib = vec![0x78, 0x01, 0x01];
    zlib.extend(len.to_le_bytes());
    zlib.extend((!len).to_le_bytes());
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &fssm {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&fssm);
    zlib.extend(((b << 16) | a).to_be_bytes());

    let mut xbch = b"XBCH".to_vec();
    xbch.extend(3u32.to_le_bytes());
    xbch.extend((zlib.len() as u64).to_le_bytes());
    xbch.extend((fssm.len() as u64).to_le_bytes());
    xbch.extend(zlib);

    let mut dxil = vec![0; 0x18];
    dxil.extend(b"XDAT");
    for field in [1, 0x24, 0x0d, 0x30, xbch.len() as u32] {
        dxil.extend(field.to_le_bytes());
    }
    dxil.extend(xbch);

    let dxil_at = 40u32;
    let private_at = (dxil_at + 8 + dxil.len() as u32).next_multiple_of(4);
    let mut bytes = b"DXBC".to_vec();
    bytes.extend([0; 16]);
    for field in [1, private_at + 9, 2, dxil_at, private_at] {
        bytes.extend(field.to_le_bytes());
    }
    bytes.extend(b"DXIL");
    bytes.extend((dxil.len() as u32).to_le_bytes());
    bytes.extend(dxil);
    bytes.resize(private_at as usize, 0);
    bytes.extend(b"PRIV");
    bytes.extend(1u32.to_le_bytes());
    bytes.push(metadata);
    bytes
}

#[test]
fn write_cache_warms_each_distinct_container_once() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("victory-translate-{}-{now}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let cache = shader_cache::open(dir.clone()).unwrap();
    let first = container(1);
    let second = container(2);
    let scan = |cache| {
        let seen = Mutex::default();
        let mut checks = Checks {
            hashes: None,
            keep: None,
            validator: None,
            host: None,
            cache,
        };
        let mut tally = Tally::default();
        for data in [&first, &second, &first] {
            translate(
                &Source::Raw { offset: 0, data },
                Path::new("synthetic"),
                &seen,
                &mut tally,
                &mut checks,
            );
        }
        tally
    };

    let cold = scan(Some(&cache));
    let warm = scan(Some(&cache));
    let survey = scan(None);
    fs::remove_dir_all(&dir).unwrap();
    assert!(cold.failed.is_empty(), "{:?}", cold.failed);
    assert_eq!((cold.cached, cold.duplicates), (2, 1));
    assert_eq!((warm.cache_hits, warm.duplicates), (2, 1));
    assert_eq!((survey.cs_emitted, survey.duplicates), (1, 2));
}

#[test]
fn filters_shared_dxil_containers_by_runtime_hash_before_deduplication() {
    let first = container(1);
    let second = container(2);
    let first_hash = blake3::hash(&first).to_hex()[..16].to_owned();
    let second_hash = blake3::hash(&second).to_hex()[..16].to_owned();
    let scan = |hashes: Option<&[String]>| {
        let seen = Mutex::default();
        let mut checks = Checks {
            hashes,
            keep: None,
            validator: None,
            host: None,
            cache: None,
        };
        let mut t = Tally::default();
        for data in [&first, &second, &second] {
            translate(
                &Source::Raw { offset: 0, data },
                Path::new("synthetic"),
                &seen,
                &mut t,
                &mut checks,
            );
        }
        (t.containers, t.cs_emitted, t.duplicates)
    };
    assert_eq!(scan(Some(&[second_hash[..12].into()])), (2, 1, 1));
    assert_eq!(scan(Some(&[first_hash, second_hash])), (3, 1, 2));
    assert_eq!(scan(Some(&["ffffffffffffffff".into()])), (0, 0, 0));
    assert_eq!(scan(Some(&[])), (0, 0, 0));
    assert_eq!(scan(None), (3, 1, 2));
}

#[test]
fn combines_command_line_and_commented_hash_file() {
    let path = std::env::temp_dir().join(format!("victory-hashes-{}.txt", std::process::id()));
    fs::write(&path, "\u{feff}# header\r\n\n Cc, DD\tee # ff\r\n11").unwrap();
    let options = HashOptions {
        hash: Some(" A, b".into()),
        hash_file: Some(path.clone()),
    };
    let parsed = options.prefixes();
    fs::remove_file(path).unwrap();
    assert_eq!(parsed.unwrap().unwrap(), ["a", "b", "cc", "dd", "ee", "11"]);
}
