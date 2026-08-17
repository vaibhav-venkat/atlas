# AGENTS

## Background

This code is written in OCaml. The package provides a semantic type layer over
tensors.

Atlas uses the `raven` tensor package and supports `rune` transformations, as well
as `Nx` where their APIs are intertwined.

The detailed roadmap is in @ROADMAP.md.



## Architecture

For all intents and purposes, this is a zero-runtime package. OCaml uses garbage
collection, so this is technically quasi-zero-runtime. The goal is to validate
semantics without running expensive numerical computations.



### Semantic Boundary

Phase 0 establishes the following contract. The exact OCaml spelling may change as
the implementation is explored, but changes must preserve these semantics.

#### Ownership

Atlas owns the **intended scientific meaning** of data:

- physical dimensions and canonical dimension algebra;
- coordinate-frame identity;
- geometric roles such as point, displacement, vector, and covector;
- semantic axis names and their order;
- lightweight, opt-in semantic kinds; and
- the meaning of those refinements through automatic differentiation.

Raven remains the sole owner of numerical representation and execution:

- dtype, device, storage, and layout;
- physical rank, shape, and extents;
- tensor primitives, broadcasting mechanics, and numerical errors; and
- the actual tensor data and computation.

Rune remains the sole owner of program transformations and AD execution. Atlas
provides a typed adapter around Rune and derives the semantics of Rune's results;
Rune does not need to know about Atlas dimensions, frames, roles, or axes.

Atlas types express **denotation and intent, not value-dependent facts**. Atlas
does not prove or encode that a particular tensor is symmetric, positive-definite,
orthogonal, normalized, nonzero, invertible, or well-conditioned. Such facts
belong in a separate, runtime-oriented validation package. A semantic label such
as `covariance` may say what data represents without claiming that Atlas verified
all numerical properties normally associated with a covariance matrix.

#### Representation and erasure

Atlas has one compositional, abstract refinement type, conceptually similar to:

```ocaml
('dimension, 'frame, 'role, 'axes, 'kind, 'data) Atlas.t
```

The parameter list and order are not yet a frozen public API. The important
properties are:

- the semantic parameters are compile-time-only;
- `'data` retains the underlying representation, primarily a Raven tensor but
  also potentially an unboxed scalar;
- specialized modules such as `Quantity`, `Geometry`, and `Axis` refine the same
  type rather than creating nested wrappers;
- the public type is abstract, so semantic claims cannot be forged accidentally;
  and
- the runtime representation is exactly that of `'data`, not a record containing
  a tensor plus semantic metadata.

Oxcaml's unboxed primitives, records, and tuples may be used where useful, but
ordinary Atlas refinements should require no evidence value or wrapper at all.

Consequently, `Atlas.unwrap x` is a representational coercion or identity
operation. It performs no tensor traversal, numerical work, allocation, or copy.
It returns the same underlying Raven value and deliberately forgets the Atlas
type refinements. Phase 1B must verify this representation claim rather than only
assuming it.

Atlas does not retain enough metadata for runtime reflection. Raw Raven
serialization therefore does not preserve dimensions, frames, roles, or axes.
Loading serialized data is a new Atlas ingress boundary. Optional schema,
formatting, and serialization helpers may accept or store separate witnesses,
but that metadata is not part of `Atlas.t`.

#### Trust boundaries

Scientific meaning cannot generally be inferred by inspecting a tensor. A Raven
`float[3]` does not reveal whether it contains metres, seconds, or a camera-frame
position. Atlas therefore uses an explicit trust model:

- named boundary constructors deliberately assign meaning and, when relevant,
  convert input units to the canonical numerical unit;
- constructors may perform inexpensive Raven-metadata checks when a claim implies
  rank or shape, but they do not inspect tensor values to prove numerical facts;
- semantics produced by trusted Atlas operations are then preserved statically;
- `Atlas.Unsafe` visibly bypasses construction or validation requirements and
  asserts a claim without justification; and
- `Atlas.Expert` exposes the low-level representational bridge required to build
  new trusted semantic abstractions and experimental algebras.

`Expert` and `Unsafe` are intentionally distinct. `Expert` is for implementing a
new abstraction with its own documented laws. `Unsafe` is for bypassing a law or
asserting that external data already satisfies a claim.

For Raven interoperability, forgetting is easy but gaining meaning is deliberate:

```text
Atlas -> Raven    explicit, free, and forgetful
Raven -> Atlas    a named boundary construction or an explicit assertion
```

Arbitrary Raven functions cannot be mapped over an Atlas value while pretending
that all refinements survive. Atlas should wrap common Raven operations only when
it can state their semantic law. Otherwise users explicitly unwrap, perform the
Raven computation, and cross an ingress boundary again.

#### Dimensions and units

Atlas types prefer **physical dimensions**, not exact measurement units. A value
is typed as length rather than metre or kilometre. Values use a canonical numerical
unit for each dimension, initially based on SI. Unit-named constructors are mainly
ingress and conversion conveniences: metres and kilometres produce the same
length type, with any necessary scaling performed as a real Raven operation.

Dimension algebra is canonical. Equality cannot depend on spelling,
parenthesization, or whether a derived name was used. For example:

```text
(length / time) * time = length
mass * acceleration    = force
force * length         = energy
length / length        = dimensionless
```

Derived dimensions such as velocity, force, and energy are aliases for canonical
base-dimension exponents, not fresh nominal types. Dimensions appear as phantom
types organized and named through modules; dimension modules and witnesses are
not stored with each value. Frames use generativity, but dimensions must not:
equal dimensions must unify globally.

The implementation may use GADTs or equality witnesses. If Oxcaml cannot express
canonical algebra ergonomically, a compile-time PPX is an acceptable implementation
path. A PPX must preserve the same canonical semantics and must not introduce
runtime dimension metadata.

Dimensionless semantic kinds such as angle, probability, count, or ratio are a
separate, lightweight refinement layer. They are optional sugar rather than the
foundation of the system. Atlas should begin with very few of them and make
explicit widening to an ordinary dimensionless value particularly easy.

Affine or logarithmic units, including Celsius and decibels, are not part of
ordinary multiplicative dimension algebra and are postponed or handled by
specialized conversion APIs.

#### Frames, geometry, and transforms

Frames are generative, compile-time-only identities:

```ocaml
module Camera = Atlas.Frame.Make ()
module World = Atlas.Frame.Make ()
```

Separate functor applications never unify. A program shares a frame by exporting
and reusing its module, not by matching a runtime name or identifier. Frame names
may aid documentation and diagnostics but are not semantic identity. Frame-neutral
data has an explicit neutral marker and does not silently unify with every frame.
Runtime-selected frame registries and existential frame graphs are postponed.

Geometric roles are erased phantom refinements. GADTs may encode relations or
witnesses internally, but Atlas values do not carry a role constructor at runtime.
The initial roles have conservative meanings:

- a point is an element of an affine space;
- a displacement translates a point and is the difference of compatible points;
- a vector is a general element of a linear space and is not automatically a
  displacement; and
- a covector is distinct from a vector and is the natural role of a gradient.

`direction` is initially postponed because it easily conflates semantic intent
with the value-dependent fact of unit normalization. Normals may later have a
distinct semantic role and dual transformation law without Atlas claiming a
normalization fact.

Role algebra is stricter than numerical tensor algebra:

```text
point - point               -> displacement
point + displacement        -> point
displacement + point        -> point
displacement + displacement -> displacement
vector + vector             -> vector
point + point               -> rejected
```

Dimensions, frames, roles, and known axes must also be compatible wherever the
operation requires compatibility. A point has no distinguished origin, so generic
negation, scaling, multiplication, and division of points are rejected even when
Raven could perform them numerically. Explicit coordinate, transform, midpoint,
or affine-combination operations may provide a meaningful interpretation.
Frame-neutral scalars may scale displacements, vectors, covectors, and ordinary
quantities when the resulting dimension is derived canonically.

A frame transform is a first-class numerical object with erased source and
destination frame parameters. It is a dimension-preserving coordinate change,
not an arbitrary linear operator. Its matrix or parameters are real Raven runtime
data; Atlas does not implement a matrix engine. Composition checks the intermediate
frame. An affine transform applies its translation to points and only its linear
part to displacements and geometric vectors. Covectors and normals use their
appropriate dual transformation law. Importing a frame transform asserts the
coordinate-change contract; Atlas does not prove numerical invertibility or
orthogonality.

General linear maps, Jacobians, and unit-changing operators are distinct from
frame transforms and may be added by the semantic linear-algebra layer.

#### Shape and semantic axes

Raven is the only authority for physical rank, extents, shape, strides, and layout.
Atlas does not duplicate shapes such as `32 x 1000 x 3` in its types and does not
attempt dependent-size arithmetic.

Atlas may encode the names and order of axes, such as `[batch; time; xyz]`. Safe
ingress may compare the number of named axes with Raven rank, and specialized
geometry construction may validate required coordinate extents using Raven
metadata. These are cheap structural checks, not numerical tensor computations.

Operations update axes only when the semantic effect is known: reductions remove
an axis, transposition reorders axes, and semantic alignment or broadcasting
introduces or aligns axes. A general reshape cannot silently preserve named axes.
Raven's ability to broadcast two tensors does not by itself make the broadcast
semantically valid.

Known semantic information is never silently weakened. Converting known axes to
unknown axes requires an explicit `forget_axes`-style operation. Values already
marked with unknown axes may remain unknown when an operation is otherwise sound.

#### Rune and automatic differentiation

Atlas supplies typed adapters such as an eventual `Atlas.Diff.grad`, `jvp`, and
`vjp`. The adapter erases Atlas refinements inside the trusted implementation,
delegates the complete numerical transformation to Rune over Raven data, and
reapplies only semantics justified by derivative rules. Directly passing the
abstract Atlas type to the unmodified `Rune.grad` spelling is not an architectural
requirement.

The initial derivative rules are:

- if `x` has dimension `X` and scalar `f x` has dimension `Y`, `grad f x` has
  dimension `Y / X`;
- a gradient is a covector in the input frame, with the input axes;
- a tangent has the same dimension, frame, and axes as its primal;
- a tangent to a point is a displacement;
- a JVP maps a valid input tangent to an output tangent carrying the output's
  dimension, frame, role relation, and axes;
- a VJP pulls an output cotangent back to an input cotangent, with dimensions
  composed according to the scalar quantity the cotangent measures; and
- a Jacobian, when supported, carries output-over-input dimensions and both the
  output and input axis structure.

In particular:

```text
E : World point[length] -> scalar[energy]
grad E : World covector[energy / length]
```

Atlas does not silently identify that covector with a force vector. Raising it to
a vector requires an explicit metric or Euclidean identification. If Atlas cannot
derive an AD result soundly, the typed adapter rejects the operation or requires
the caller to forget the relevant semantics explicitly.

#### Strictness and extensibility

No Atlas operation silently weakens known dimensions, frames, roles, kinds, or
axes. It either derives a justified result or fails to type-check. Explicit
forgetting is always permitted, and raw numerical work remains available through
`unwrap`.

Atlas provides a strict default algebra but is not a closed mathematical universe.
Frames, axes, semantic kinds, derived dimensions, and experimental meanings can
be defined outside core. Core generic operations need support only the laws Atlas
officially guarantees; extensions may provide their own phantom semantics,
witnesses, axioms, and operations through `Atlas.Expert`. The design must avoid
closed encodings that make such experiments impossible. Experimental systems can
coexist without globally modifying `Atlas.add` or `Atlas.sub`.

The built-in core remains deliberately small. Domain semantics such as meshes,
robotics-specific concepts, uncertainty, and similar taxonomies belong in
extensions until repeated use justifies promotion.

#### Phase 0 exit test

Phase 0 is complete when the project can maintain this explanation:

1. `Atlas.unwrap x` is essentially free because `Atlas.t` has the same runtime
   representation as its underlying `'data`; unwrapping only changes the static
   view and performs no allocation, copy, traversal, validation, or numerical work.
2. Dimensions, frames, roles, axes, semantic kinds, and AD variance are
   compile-time-only refinements.
3. Raven tensors, scalars, transform matrices, and ordinary numerical results are
   runtime data. Unit conversions and Rune/Raven computations are real numerical
   work, as they must be.
4. Boundary shape checks may inspect Raven metadata, but Atlas never maintains a
   duplicate runtime semantic record and never validates value-dependent facts.
5. Moving from Atlas to Raven forgets meaning; moving from Raven to Atlas is an
   explicit trust boundary.

This contract satisfies the Phase 0 exit condition. Phase 1 must test the chosen
Oxcaml representation, and Phase 1B must measure the zero-cost claim on generated
code and allocation behavior.
