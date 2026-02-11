#pragma once
#include <concepts>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

template <typename T>
concept Streamable = requires(T t, std::ostream& os) {
  { os << t } -> std::convertible_to<std::ostream&>;
};

struct FloatFormat {
  int precision = std::numeric_limits<double>::max_digits10;
  bool fixed = true;
};

class CSVWriter {
 public:
  explicit CSVWriter(const std::filesystem::path& path, char delimiter = ',',
                     size_t buffer_rows = 1000,
                     FloatFormat float_fmt = FloatFormat{})
      : delimiter_(delimiter),
        buffer_rows_(buffer_rows),
        float_format_(float_fmt),
        file_(path, std::ios::binary) {
    if (!file_) {
      throw std::runtime_error("Failed to open file: " + path.string());
    }
    buffer_.reserve(buffer_rows);
  }

  void write_header(const std::vector<std::string>& headers) {
    write_row(headers);
  }

  template <Streamable T>
  void write_row(const std::vector<T>& row) {
    buffer_.emplace_back();
    buffer_.back().reserve(row.size());
    for (const auto& field : row) {
      buffer_.back().push_back(to_string(field));
    }

    if (buffer_.size() >= buffer_rows_) {
      flush();
    }
  }

  template <Streamable... Args>
  void write_row(const Args&... args) {
    buffer_.emplace_back();
    buffer_.back().reserve(sizeof...(Args));
    (buffer_.back().push_back(to_string(args)), ...);

    if (buffer_.size() >= buffer_rows_) {
      flush();
    }
  }

  template <Streamable T>
  void write_row(std::initializer_list<T> row) {
    write_row(std::vector<T>(row));
  }

  void flush() {
    for (const auto& row : buffer_) {
      write_row_to_file(row);
    }
    buffer_.clear();
    file_.flush();
  }

  ~CSVWriter() {
    try {
      flush();
    } catch (...) {
    }
  }

 private:
  char delimiter_ = ',';
  size_t buffer_rows_;
  FloatFormat float_format_;
  std::ofstream file_;
  std::vector<std::vector<std::string>> buffer_;

  template <Streamable T>
  std::string to_string(const T& value) const {
    if constexpr (std::is_same_v<T, std::string>) {
      return value;
    } else if constexpr (std::is_same_v<T, const char*> ||
                         std::is_same_v<T, char*>) {
      return std::string(value);
    } else if constexpr (std::is_same_v<T, std::string_view>) {
      return std::string(value);
    } else if constexpr (std::is_same_v<T, bool>) {
      return value ? "true" : "false";
    } else if constexpr (std::is_floating_point_v<T>) {
      std::ostringstream oss;
      if (float_format_.fixed) {
        oss << std::fixed;
      }
      oss << std::setprecision(float_format_.precision) << value;
      return oss.str();
    } else {
      std::ostringstream oss;
      oss << value;
      return oss.str();
    }
  }

  void write_row_to_file(const std::vector<std::string>& row) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i > 0) {
        file_ << delimiter_;
      }
      write_field(row[i]);
    }
    file_ << '\n';
  }

  void write_field(const std::string& field) {
    bool needs_quotes = needs_quoting(field);

    if (needs_quotes) {
      file_ << '"';
      for (char c : field) {
        if (c == '"') {
          file_ << "\"\"";
        } else {
          file_ << c;
        }
      }
      file_ << '"';
    } else {
      file_ << field;
    }
  }

  bool needs_quoting(const std::string& field) const {
    for (char c : field) {
      if (c == delimiter_ || c == '"' || c == '\n' || c == '\r') {
        return true;
      }
    }
    return false;
  }
};
