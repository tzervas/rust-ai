//! Scientific color schemes for data visualization
//!
//! Provides perceptually uniform colormaps commonly used in scientific visualization:
//! - Viridis: Blue-green-yellow, perceptually uniform
//! - Plasma: Magenta-orange-yellow
//! - Magma: Black-purple-orange-white
//! - Inferno: Black-purple-red-yellow
//! - Cividis: Blue-yellow, optimized for color blindness

use std::f32::consts::PI;

/// RGBA color with f32 components in [0, 1] range
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    /// Create a new color from RGB values (alpha defaults to 1.0)
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create a new color from RGBA values
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create color from u8 RGB values (0-255)
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: 1.0,
        }
    }

    /// Create color from hex string (e.g., "#FF5500" or "FF5500")
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        Some(Self::from_u8(r, g, b))
    }

    /// Convert to [r, g, b, a] array
    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Convert to [r, g, b] array (ignoring alpha)
    pub fn to_rgb_array(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Linear interpolation between two colors
    pub fn lerp(&self, other: &Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// White
    pub const WHITE: Color = Color::rgb(1.0, 1.0, 1.0);

    /// Black
    pub const BLACK: Color = Color::rgb(0.0, 0.0, 0.0);

    /// Red
    pub const RED: Color = Color::rgb(1.0, 0.0, 0.0);

    /// Green
    pub const GREEN: Color = Color::rgb(0.0, 1.0, 0.0);

    /// Blue
    pub const BLUE: Color = Color::rgb(0.0, 0.0, 1.0);
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

/// Scientific colormap for mapping scalar values to colors
#[derive(Debug, Clone)]
pub struct Colormap {
    /// Name of the colormap
    pub name: &'static str,
    /// Control points for the colormap (value, color pairs)
    control_points: Vec<(f32, Color)>,
}

impl Colormap {
    /// Create a new colormap from control points
    /// Points should be sorted by value in [0, 1] range
    pub fn new(name: &'static str, points: Vec<(f32, Color)>) -> Self {
        Self {
            name,
            control_points: points,
        }
    }

    /// Map a scalar value [0, 1] to a color
    pub fn map(&self, value: f32) -> Color {
        let value = value.clamp(0.0, 1.0);

        if self.control_points.is_empty() {
            return Color::WHITE;
        }

        if self.control_points.len() == 1 {
            return self.control_points[0].1;
        }

        // Find the two control points to interpolate between
        for i in 0..self.control_points.len() - 1 {
            let (v0, c0) = &self.control_points[i];
            let (v1, c1) = &self.control_points[i + 1];

            if value >= *v0 && value <= *v1 {
                let t = if (v1 - v0).abs() < 1e-6 {
                    0.0
                } else {
                    (value - v0) / (v1 - v0)
                };
                return c0.lerp(c1, t);
            }
        }

        // Value out of range, return closest endpoint
        if value <= self.control_points[0].0 {
            self.control_points[0].1
        } else {
            self.control_points.last().unwrap().1
        }
    }

    /// Map a value from an arbitrary range to a color
    pub fn map_range(&self, value: f32, min: f32, max: f32) -> Color {
        if (max - min).abs() < 1e-10 {
            return self.map(0.5);
        }
        let normalized = (value - min) / (max - min);
        self.map(normalized)
    }

    /// Generate N discrete colors evenly spaced in the colormap
    pub fn discrete(&self, n: usize) -> Vec<Color> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![self.map(0.5)];
        }
        (0..n)
            .map(|i| self.map(i as f32 / (n - 1) as f32))
            .collect()
    }
}

/// Viridis colormap - perceptually uniform blue-green-yellow
/// Designed by van der Walt, Smith, and Firing for matplotlib
pub fn viridis() -> Colormap {
    Colormap::new(
        "viridis",
        vec![
            (0.00, Color::from_hex("#440154").unwrap()),
            (0.10, Color::from_hex("#482878").unwrap()),
            (0.20, Color::from_hex("#3E4A89").unwrap()),
            (0.30, Color::from_hex("#31688E").unwrap()),
            (0.40, Color::from_hex("#26828E").unwrap()),
            (0.50, Color::from_hex("#1F9E89").unwrap()),
            (0.60, Color::from_hex("#35B779").unwrap()),
            (0.70, Color::from_hex("#6DCD59").unwrap()),
            (0.80, Color::from_hex("#B4DE2C").unwrap()),
            (0.90, Color::from_hex("#FDE725").unwrap()),
            (1.00, Color::from_hex("#FDE725").unwrap()),
        ],
    )
}

/// Plasma colormap - perceptually uniform magenta-orange-yellow
pub fn plasma() -> Colormap {
    Colormap::new(
        "plasma",
        vec![
            (0.00, Color::from_hex("#0D0887").unwrap()),
            (0.10, Color::from_hex("#41049D").unwrap()),
            (0.20, Color::from_hex("#6A00A8").unwrap()),
            (0.30, Color::from_hex("#8F0DA4").unwrap()),
            (0.40, Color::from_hex("#B12A90").unwrap()),
            (0.50, Color::from_hex("#CC4778").unwrap()),
            (0.60, Color::from_hex("#E16462").unwrap()),
            (0.70, Color::from_hex("#F2844B").unwrap()),
            (0.80, Color::from_hex("#FCA636").unwrap()),
            (0.90, Color::from_hex("#FCCE25").unwrap()),
            (1.00, Color::from_hex("#F0F921").unwrap()),
        ],
    )
}

/// Magma colormap - perceptually uniform black-purple-orange-white
pub fn magma() -> Colormap {
    Colormap::new(
        "magma",
        vec![
            (0.00, Color::from_hex("#000004").unwrap()),
            (0.10, Color::from_hex("#180F3D").unwrap()),
            (0.20, Color::from_hex("#440F76").unwrap()),
            (0.30, Color::from_hex("#721F81").unwrap()),
            (0.40, Color::from_hex("#9E2F7F").unwrap()),
            (0.50, Color::from_hex("#CD4071").unwrap()),
            (0.60, Color::from_hex("#F1605D").unwrap()),
            (0.70, Color::from_hex("#FD9668").unwrap()),
            (0.80, Color::from_hex("#FECA8D").unwrap()),
            (0.90, Color::from_hex("#FCFDBF").unwrap()),
            (1.00, Color::from_hex("#FCFDBF").unwrap()),
        ],
    )
}

/// Inferno colormap - perceptually uniform black-purple-red-yellow
pub fn inferno() -> Colormap {
    Colormap::new(
        "inferno",
        vec![
            (0.00, Color::from_hex("#000004").unwrap()),
            (0.10, Color::from_hex("#1B0C41").unwrap()),
            (0.20, Color::from_hex("#4A0C6B").unwrap()),
            (0.30, Color::from_hex("#781C6D").unwrap()),
            (0.40, Color::from_hex("#A52C60").unwrap()),
            (0.50, Color::from_hex("#CF4446").unwrap()),
            (0.60, Color::from_hex("#ED6925").unwrap()),
            (0.70, Color::from_hex("#FB9A06").unwrap()),
            (0.80, Color::from_hex("#F7D03C").unwrap()),
            (0.90, Color::from_hex("#FCFFA4").unwrap()),
            (1.00, Color::from_hex("#FCFFA4").unwrap()),
        ],
    )
}

/// Cividis colormap - optimized for color vision deficiency
pub fn cividis() -> Colormap {
    Colormap::new(
        "cividis",
        vec![
            (0.00, Color::from_hex("#00224E").unwrap()),
            (0.10, Color::from_hex("#123570").unwrap()),
            (0.20, Color::from_hex("#3B496C").unwrap()),
            (0.30, Color::from_hex("#575D6D").unwrap()),
            (0.40, Color::from_hex("#707173").unwrap()),
            (0.50, Color::from_hex("#8A8678").unwrap()),
            (0.60, Color::from_hex("#A59C74").unwrap()),
            (0.70, Color::from_hex("#C3B369").unwrap()),
            (0.80, Color::from_hex("#E1CC55").unwrap()),
            (0.90, Color::from_hex("#FDE838").unwrap()),
            (1.00, Color::from_hex("#FDE838").unwrap()),
        ],
    )
}

/// Coolwarm diverging colormap - blue (cool) to red (warm)
/// Good for showing deviations from a center point
pub fn coolwarm() -> Colormap {
    Colormap::new(
        "coolwarm",
        vec![
            (0.00, Color::from_hex("#3B4CC0").unwrap()),
            (0.10, Color::from_hex("#5A78D1").unwrap()),
            (0.20, Color::from_hex("#7B9FE2").unwrap()),
            (0.30, Color::from_hex("#9EBFF0").unwrap()),
            (0.40, Color::from_hex("#C0D4F5").unwrap()),
            (0.50, Color::from_hex("#DDDDDD").unwrap()),
            (0.60, Color::from_hex("#F2C9B4").unwrap()),
            (0.70, Color::from_hex("#EFA98A").unwrap()),
            (0.80, Color::from_hex("#E57A5E").unwrap()),
            (0.90, Color::from_hex("#D44D3C").unwrap()),
            (1.00, Color::from_hex("#B40426").unwrap()),
        ],
    )
}

/// Scientific colormap preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColormapPreset {
    #[default]
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Cividis,
    Coolwarm,
}

impl ColormapPreset {
    /// Get the colormap for this preset
    pub fn colormap(&self) -> Colormap {
        match self {
            ColormapPreset::Viridis => viridis(),
            ColormapPreset::Plasma => plasma(),
            ColormapPreset::Magma => magma(),
            ColormapPreset::Inferno => inferno(),
            ColormapPreset::Cividis => cividis(),
            ColormapPreset::Coolwarm => coolwarm(),
        }
    }
}

/// Color palette for categorical data
#[derive(Debug, Clone)]
pub struct CategoricalPalette {
    colors: Vec<Color>,
}

impl CategoricalPalette {
    /// Create a palette from a list of colors
    pub fn new(colors: Vec<Color>) -> Self {
        Self { colors }
    }

    /// Get color by index (wraps around if index exceeds palette size)
    pub fn get(&self, index: usize) -> Color {
        if self.colors.is_empty() {
            return Color::WHITE;
        }
        self.colors[index % self.colors.len()]
    }

    /// Number of colors in the palette
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Check if palette is empty
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

/// Tab10 categorical palette (matplotlib default)
pub fn tab10() -> CategoricalPalette {
    CategoricalPalette::new(vec![
        Color::from_hex("#1F77B4").unwrap(), // Blue
        Color::from_hex("#FF7F0E").unwrap(), // Orange
        Color::from_hex("#2CA02C").unwrap(), // Green
        Color::from_hex("#D62728").unwrap(), // Red
        Color::from_hex("#9467BD").unwrap(), // Purple
        Color::from_hex("#8C564B").unwrap(), // Brown
        Color::from_hex("#E377C2").unwrap(), // Pink
        Color::from_hex("#7F7F7F").unwrap(), // Gray
        Color::from_hex("#BCBD22").unwrap(), // Olive
        Color::from_hex("#17BECF").unwrap(), // Cyan
    ])
}

/// Generate rainbow colors for N categories
pub fn rainbow(n: usize) -> CategoricalPalette {
    let colors: Vec<Color> = (0..n)
        .map(|i| {
            let hue = i as f32 / n as f32;
            hsv_to_rgb(hue, 0.8, 0.9)
        })
        .collect();
    CategoricalPalette::new(colors)
}

/// Convert HSV to RGB color
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    Color::rgb(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let c = Color::from_hex("#FF5500").unwrap();
        assert!((c.r - 1.0).abs() < 0.01);
        assert!((c.g - 0.333).abs() < 0.01);
        assert!((c.b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_colormap_endpoints() {
        let cmap = viridis();
        let start = cmap.map(0.0);
        let end = cmap.map(1.0);
        // Viridis starts dark purple, ends yellow
        assert!(start.r < 0.3);
        assert!(end.g > 0.8);
    }

    #[test]
    fn test_colormap_interpolation() {
        let cmap = viridis();
        let mid = cmap.map(0.5);
        // Middle of viridis is a teal color
        assert!(mid.g > 0.5);
    }

    #[test]
    fn test_categorical_wrap() {
        let palette = tab10();
        let c0 = palette.get(0);
        let c10 = palette.get(10);
        assert_eq!(c0.r, c10.r);
    }
}
