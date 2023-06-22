
# [0.2.0] - 2023-06-22
### Added
- `IdContainer` trait for storing IDs in MinHashIndex, and three implementations `HashSetContainer`, 
`VecContainer` and `SmallVecContainer`
- Method `query_one` to `SimHashIndex` 
- Criterion benchmarks

### Changed
- MinHashIndex `remove` methods return existing signature

### Fixed
- Fix issue when duplicate ids may cause a crash. (#19)


