pub fn load_cache(path: &str) -> String {
    std::fs::read_to_string(path).unwrap()
}
