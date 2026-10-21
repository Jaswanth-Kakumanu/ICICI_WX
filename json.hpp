// json.hpp - a tiny JSON writer so the project needs no external libraries.
//
//   JsonWriter j;
//   j.beginObject();
//     j.field("name", "demo");
//     j.key("values"); j.array(std::vector<double>{1, 2, 3});
//   j.endObject();
//   std::string text = j.str();
#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fx {

class JsonWriter {
public:
    JsonWriter() { os_ << std::setprecision(10); }

    void beginObject() { sep(); os_ << '{'; first_.push_back(true); }
    void endObject() { os_ << '}'; first_.pop_back(); }
    void beginArray() { sep(); os_ << '['; first_.push_back(true); }
    void endArray() { os_ << ']'; first_.pop_back(); }

    void key(const std::string& k) {
        sep();
        writeString(k);
        os_ << ':';
        afterKey_ = true;
    }

    void value(double v) {
        sep();
        if (std::isfinite(v)) os_ << v;
        else os_ << "null";  // NaN / infinity are not valid JSON
    }
    void value(int v) { sep(); os_ << v; }
    void value(long v) { sep(); os_ << v; }
    void value(const std::string& v) { sep(); writeString(v); }
    void value(const char* v) { value(std::string(v)); }

    void array(const std::vector<double>& v) {
        beginArray();
        for (double x : v) value(x);
        endArray();
    }
    void array(const std::vector<std::string>& v) {
        beginArray();
        for (const auto& x : v) value(x);
        endArray();
    }
    void array(const std::vector<int>& v) {
        beginArray();
        for (int x : v) value(x);
        endArray();
    }

    template <class T>
    void field(const std::string& k, const T& v) {
        key(k);
        value(v);
    }

    std::string str() const { return os_.str(); }

private:
    // Emits the comma between siblings (but not directly after a key).
    void sep() {
        if (afterKey_) {
            afterKey_ = false;
            return;
        }
        if (!first_.empty()) {
            if (!first_.back()) os_ << ',';
            first_.back() = false;
        }
    }

    void writeString(const std::string& s) {
        os_ << '"';
        for (char ch : s) {
            switch (ch) {
                case '"': os_ << "\\\""; break;
                case '\\': os_ << "\\\\"; break;
                case '\n': os_ << "\\n"; break;
                default: os_ << ch;
            }
        }
        os_ << '"';
    }

    std::ostringstream os_;
    std::vector<bool> first_;
    bool afterKey_ = false;
};

}  // namespace fx
