
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Deserialize, Clone)]
pub struct PatientRecord {
    pub age: f32,
    pub sex: f32,
    pub cp: f32,
    pub trestbps: f32,
    pub chol: f32,
    pub fbs: f32,
    pub restecg: f32,
    pub thalach: f32,
    pub exang: f32,
    pub oldpeak: f32,
    pub slope: f32,
    pub ca: String,
    pub thal: String,
    pub num: u8,
}

#[derive(Debug, Clone)]
pub struct ProcessedPatientRecord {
    pub features: Vec<f32>,
    pub target: u8,
}

fn clean_and_convert(record: PatientRecord) -> Option<ProcessedPatientRecord> {
    let ca = record.ca.trim();
    let thal = record.thal.trim();

    if ca == "?" || thal == "?" {
        return None;
    }

    let ca_val: f32 = ca.parse().ok()?;
    let thal_val: f32 = thal.parse().ok()?;

    Some(ProcessedPatientRecord {
        features: vec![
            record.age,
            record.sex,
            record.cp,
            record.trestbps,
            record.chol,
            record.fbs,
            record.restecg,
            record.thalach,
            record.exang,
            record.oldpeak,
            record.slope,
            ca_val,
            thal_val,
        ],
        target: if record.num > 0 { 1 } else { 0 },
    })
}

pub fn load_and_preprocess_data(path: &str) -> Result<Vec<ProcessedPatientRecord>, std::io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(contents.as_bytes());

    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: PatientRecord = result?;
        if let Some(processed_record) = clean_and_convert(record) {
            records.push(processed_record);
        }
    }
    Ok(records)
}

pub fn train_test_split(
    data: &mut Vec<ProcessedPatientRecord>,
    test_size: f32,
) -> (Vec<ProcessedPatientRecord>, Vec<ProcessedPatientRecord>) {
    data.shuffle(&mut thread_rng());
    let test_count = (data.len() as f32 * test_size).round() as usize;
    let test_set = data.drain(..test_count).collect();
    let train_set = data.drain(..).collect();
    (train_set, test_set)
}

