### Changed

- **The conversions of floating point in the PME templates are one op, `mdrt.template_convert`**, which the compiler makes an extension, a truncation, or a cast of the bits in IR for the types of the instance, on the host and on a device; the seven text placeholders are gone (D370, #739). The lowered programs do not change (`docs/pme.md`, Section 3).
