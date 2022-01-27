#include "mold.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

namespace mold::elf {

using E = RISCV64;

static u32 bit(u32 val, i64 pos) {
  return (val & (1 << pos)) ? 1 : 0;
}

// Returns [hi:lo] bits of val.
static u32 bits(u32 val, i64 hi, i64 lo) {
  return (val >> lo) & ((1 << (hi - lo + 1)) - 1);
}

static u32 itype(u32 val) {
  return val << 19;
}

static u32 stype(u32 val) {
  return bits(val, 11, 5) << 24 | bits(val, 4, 0) << 6;
}

static u32 btype(u32 val) {
  return bit(val, 12) << 31 | bits(val, 10, 5) << 25 |
         bits(val, 4, 1) << 8 | bit(val, 11) << 7;
}

static u32 utype(u32 val) {
  return bits(val, 31, 12) << 11;
}

static u32 jtype(u32 val) {
  return bit(val, 20) << 30 | bits(val, 10, 1) << 20 |
         bit(val, 11) << 19 | bits(val, 19, 12) << 11;
}

static inline u32 get_bits16(u32 val, i64 *layout) {
  u32 ret = 0;
#pragma unroll
#pragma GCC unroll 16
  for (int i = 0; i < 16; i++)
    if (layout[i] != -1)
      ret |= bit(val, layout[i]) << (16 - i);
  return ret;
}

static u32 cbtype(u32 val) {
  i64 layout[] = {-1, -1, -1, 8, 4, 3, -1, -1, -1, 7, 6, 2, 1, 5, -1, -1};
  return get_bits16(val, layout);
}

static u32 cjtype(u32 val) {
  i64 layout[] = {-1, -1, -1, 11, 4, 9, 8, 10, 6, 7, 3, 2, 1, 5, -1, -1};
  return get_bits16(val, layout);
}

static void write_plt_header(Context<E> &ctx) {
  u32 *buf = (u32 *)(ctx.buf + ctx.plt->shdr.sh_offset);

  static const u32 plt0[] = {
    0x00000397, // auipc  t2, %pcrel_hi(.got.plt)
    0x41c30333, // sub    t1, t1, t3               # .plt entry + hdr + 12
    0x0003be03, // ld     t3, %pcrel_lo(1b)(t2)    # _dl_runtime_resolve
    0xfd430313, // addi   t1, t1, -44              # .plt entry
    0x00038293, // addi   t0, t2, %pcrel_lo(1b)    # &.got.plt
    0x00135313, // srli   t1, t1, 1                # .plt entry offset
    0x0082b283, // ld     t0, 8(t0)                # link map
    0x000e0067, // jr     t3
  };

  u64 gotplt = ctx.gotplt->shdr.sh_addr;
  u64 plt = ctx.plt->shdr.sh_addr;

  memcpy(buf, plt0, sizeof(plt0));
  buf[0] |= utype(gotplt - plt);
  buf[2] |= itype(gotplt - plt);
  buf[4] |= itype(gotplt - plt);
}

static void write_plt_entry(Context<E> &ctx, Symbol<E> &sym) {
  u32 *ent = (u32 *)(ctx.buf + ctx.plt->shdr.sh_offset + ctx.plt_hdr_size +
                     sym.get_plt_idx(ctx) * ctx.plt_size);

  static const u32 data[] = {
    0x00000e17, // auipc   t3, %pcrel_hi(function@.got.plt)
    0x000e3e03, // ld      t3, %pcrel_lo(1b)(t3)
    0x000e0367, // jalr    t1, t3
    0x00000013, // nop
  };

  u64 gotplt = sym.get_gotplt_addr(ctx);
  u64 plt = sym.get_plt_addr(ctx);

  memcpy(ent, data, sizeof(data));
  ent[0] |= utype(gotplt - plt);
  ent[1] |= itype(gotplt - plt);
}

template <>
void PltSection<E>::copy_buf(Context<E> &ctx) {
  write_plt_header(ctx);
  for (Symbol<E> *sym : symbols)
    write_plt_entry(ctx, *sym);
}

template <>
void PltGotSection<E>::copy_buf(Context<E> &ctx) {
  u32 *buf = (u32 *)(ctx.buf + ctx.plt->shdr.sh_offset);

  static const u32 data[] = {
    0x00000e17, // auipc   t3, %pcrel_hi(function@.got.plt)
    0x000e3e03, // ld      t3, %pcrel_lo(1b)(t3)
    0x000e0367, // jalr    t1, t3
    0x00000013, // nop
  };

  for (Symbol<E> *sym : symbols) {
    u32 *ent = buf + sym->get_pltgot_idx(ctx) * 4;
    u64 got = sym->get_got_addr(ctx);
    u64 plt = sym->get_plt_addr(ctx);

    memcpy(ent, data, sizeof(data));
    ent[0] |= utype(got - plt);
    ent[1] |= itype(got - plt);
  }
}

template <>
void EhFrameSection<E>::apply_reloc(Context<E> &ctx, ElfRel<E> &rel,
                                    u64 offset, u64 val) {
  u8 *loc = ctx.buf + this->shdr.sh_offset + offset;

  switch (rel.r_type) {
  case R_RISCV_ADD32:
    *(u32 *)loc += val;
    return;
  case R_RISCV_SUB8:
    *loc -= val;
    return;
  case R_RISCV_SUB16:
    *(u16 *)loc -= val;
    return;
  case R_RISCV_SUB32:
    *(u32 *)loc -= val;
    return;
  case R_RISCV_SUB6:
    *loc = (*loc - val) & 0b11'1111;
    return;
  case R_RISCV_SET6:
    *loc = (*loc + val) & 0b11'1111;
    return;
  case R_RISCV_SET8:
    *loc = val;
    return;
  case R_RISCV_SET16:
    *(u16 *)loc = val;
    return;
  case R_RISCV_32_PCREL:
    *(u32 *)loc = val - this->shdr.sh_addr - offset;
    return;
  }
  Fatal(ctx) << "unsupported relocation in .eh_frame: " << rel;
}

template <>
void InputSection<E>::apply_reloc_alloc(Context<E> &ctx, u8 *base) {
  ElfRel<E> *dynrel = nullptr;
  std::span<ElfRel<E>> rels = get_rels(ctx);

  i64 frag_idx = 0;

  if (ctx.reldyn)
    dynrel = (ElfRel<E> *)(ctx.buf + ctx.reldyn->shdr.sh_offset +
                           file.reldyn_offset + this->reldyn_offset);

  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];
    if (rel.r_type == R_RISCV_NONE)
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym];
    u8 *loc = base + rel.r_offset;

    const SectionFragmentRef<E> *frag_ref = nullptr;
    if (rel_fragments && rel_fragments[frag_idx].idx == i)
      frag_ref = &rel_fragments[frag_idx++];

    auto overflow_check = [&](i64 val, i64 lo, i64 hi) {
      if (val < lo || hi <= val)
        Error(ctx) << *this << ": relocation " << rel << " against "
                   << sym << " out of range: " << val << " is not in ["
                   << lo << ", " << hi << ")";
    };

#define S   (frag_ref ? frag_ref->frag->get_addr(ctx) : sym.get_addr(ctx))
#define A   (frag_ref ? frag_ref->addend : rel.r_addend)
#define P   (output_section->shdr.sh_addr + offset + rel.r_offset)
#define G   (sym.get_got_addr(ctx) - ctx.got->shdr.sh_addr)
#define GOT ctx.got->shdr.sh_addr

    if (needs_dynrel[i]) {
      *dynrel++ = {P, R_RISCV_64, (u32)sym.get_dynsym_idx(ctx), A};
      *(u64 *)loc = A;
      continue;
    }

    if (needs_baserel[i]) {
      if (!is_relr_reloc(ctx, rel))
        *dynrel++ = {P, R_RISCV_RELATIVE, 0, (i64)(S + A)};
      *(u64 *)loc = S + A;
      continue;
    }

    switch (rel.r_type) {
    case R_RISCV_32:
      *(u32 *)loc = S + A;
      break;
    case R_RISCV_64:
      *(u64 *)loc = S + A;
      break;
    case R_RISCV_TLS_DTPMOD32:
    case R_RISCV_TLS_DTPMOD64:
    case R_RISCV_TLS_DTPREL32:
    case R_RISCV_TLS_DTPREL64:
    case R_RISCV_TLS_TPREL32:
    case R_RISCV_TLS_TPREL64:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_RISCV_BRANCH:
      *(u32 *)loc |= btype(S + A - P);
      break;
    case R_RISCV_JAL:
      *(u32 *)loc |= jtype(S + A - P);
      break;
    case R_RISCV_CALL:
    case R_RISCV_CALL_PLT:
      *(u32 *)loc |= utype(S + A - P);
      *(u32 *)(loc + 4) |= jtype(S + A - P);
      break;
    case R_RISCV_GOT_HI20:
      *(u32 *)loc = G + A - P;
      break;
    case R_RISCV_TLS_GOT_HI20:
    case R_RISCV_TLS_GD_HI20:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_RISCV_PCREL_HI20:
      *(u32 *)loc = S + A - P;
    case R_RISCV_PCREL_LO12_I:
    case R_RISCV_LO12_I:
    case R_RISCV_TPREL_LO12_I:
      *(u32 *)loc |= itype(*(base + sym.value));
      break;
    case R_RISCV_PCREL_LO12_S:
    case R_RISCV_LO12_S:
    case R_RISCV_TPREL_LO12_S:
      *(u32 *)loc |= stype(*(base + sym.value));
      break;
    case R_RISCV_HI20:
      *(u32 *)loc = S + A;
      break;
    case R_RISCV_TPREL_HI20:
      *(u32 *)loc = S + A - ctx.tls_begin;
      break;
    case R_RISCV_TPREL_ADD:
      break;
    case R_RISCV_ADD8:
      loc += S + A;
      break;
    case R_RISCV_ADD16:
      *(u16 *)loc += S + A;
      break;
    case R_RISCV_ADD32:
      *(u32 *)loc += S + A;
      break;
    case R_RISCV_ADD64:
      *(u64 *)loc += S + A;
      break;
    case R_RISCV_SUB8:
      loc -= S + A;
      break;
    case R_RISCV_SUB16:
      *(u16 *)loc -= S + A;
      break;
    case R_RISCV_SUB32:
      *(u32 *)loc -= S + A;
      break;
    case R_RISCV_SUB64:
      *(u64 *)loc -= S + A;
      break;
    case R_RISCV_ALIGN:
      break;
    case R_RISCV_RVC_BRANCH:
      *(u16 *)loc = cbtype(S + A - P);
      break;
    case R_RISCV_RVC_JUMP:
      *(u16 *)loc = cjtype(S + A - P);
      break;
    case R_RISCV_RVC_LUI:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_RISCV_RELAX:
      break;
    case R_RISCV_SUB6:
    case R_RISCV_SET6:
    case R_RISCV_SET8:
    case R_RISCV_SET16:
    case R_RISCV_SET32:
    case R_RISCV_32_PCREL:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    default:
      Error(ctx) << *this << ": unknown relocation: " << rel;
    }

#undef S
#undef A
#undef P
#undef G
#undef GOT
  }

  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &r = rels[i];
    u32 *loc = (u32 *)(base + r.r_offset);

    switch (r.r_type) {
    case R_RISCV_GOT_HI20:
    case R_RISCV_HI20:
    case R_RISCV_TPREL_HI20:
      *loc = *(u32 *)&contents[r.r_offset] | utype(*loc);
      break;
    }
  }
}

template <>
void InputSection<E>::apply_reloc_nonalloc(Context<E> &ctx, u8 *base) {}

template <>
void InputSection<E>::scan_relocations(Context<E> &ctx) {
  assert(shdr.sh_flags & SHF_ALLOC);

  this->reldyn_offset = file.num_dynrel * sizeof(ElfRel<E>);
  std::span<ElfRel<E>> rels = get_rels(ctx);

  // Scan relocations
  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];
    if (rel.r_type == R_RISCV_NONE)
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym];

    if (!sym.file) {
      report_undef(ctx, sym);
      continue;
    }

    if (sym.get_type() == STT_GNU_IFUNC) {
      sym.flags |= NEEDS_GOT;
      sym.flags |= NEEDS_PLT;
    }

    switch (rel.r_type) {
    case R_RISCV_32:
    case R_RISCV_HI20: {
      Action table[][4] = {
        // Absolute  Local    Imported data  Imported code
        {  NONE,     NONE,    ERROR,         ERROR },      // DSO
        {  NONE,     NONE,    COPYREL,       PLT   },      // PIE
        {  NONE,     NONE,    COPYREL,       PLT   },      // PDE
      };
      dispatch(ctx, table, i, rel, sym);
      break;
    }
    case R_RISCV_64: {
      Action table[][4] = {
        // Absolute  Local    Imported data  Imported code
        {  NONE,     BASEREL, DYNREL,        DYNREL },     // DSO
        {  NONE,     BASEREL, DYNREL,        DYNREL },     // PIE
        {  NONE,     NONE,    COPYREL,       PLT    },     // PDE
      };
      dispatch(ctx, table, i, rel, sym);
      break;
    }
    case R_RISCV_TLS_DTPMOD32:
    case R_RISCV_TLS_DTPMOD64:
    case R_RISCV_TLS_DTPREL32:
    case R_RISCV_TLS_DTPREL64:
    case R_RISCV_TLS_TPREL32:
    case R_RISCV_TLS_TPREL64:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_RISCV_BRANCH:
    case R_RISCV_JAL:
      break;
    case R_RISCV_CALL:
    case R_RISCV_CALL_PLT:
      if (sym.is_imported)
        sym.flags |= NEEDS_PLT;
      break;
    case R_RISCV_GOT_HI20:
      sym.flags |= NEEDS_GOT;
      break;
    case R_RISCV_TLS_GOT_HI20:
    case R_RISCV_TLS_GD_HI20:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_RISCV_PCREL_HI20:
    case R_RISCV_PCREL_LO12_I:
    case R_RISCV_PCREL_LO12_S:
    case R_RISCV_LO12_I:
    case R_RISCV_LO12_S:
    case R_RISCV_TPREL_HI20:
    case R_RISCV_TPREL_LO12_I:
    case R_RISCV_TPREL_LO12_S:
    case R_RISCV_TPREL_ADD:
    case R_RISCV_ADD8:
    case R_RISCV_ADD16:
    case R_RISCV_ADD32:
    case R_RISCV_ADD64:
    case R_RISCV_SUB8:
    case R_RISCV_SUB16:
    case R_RISCV_SUB32:
    case R_RISCV_SUB64:
    case R_RISCV_ALIGN:
      break;
    case R_RISCV_RVC_BRANCH:
    case R_RISCV_RVC_JUMP:
      break;
    case R_RISCV_RVC_LUI:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_RISCV_RELAX:
      break;
    case R_RISCV_SUB6:
    case R_RISCV_SET6:
    case R_RISCV_SET8:
    case R_RISCV_SET16:
    case R_RISCV_SET32:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_RISCV_32_PCREL: {
      Action table[][4] = {
        // Absolute  Local  Imported data  Imported code
        {  ERROR,    NONE,  ERROR,         ERROR },      // DSO
        {  ERROR,    NONE,  COPYREL,       PLT   },      // PIE
        {  NONE,     NONE,  COPYREL,       PLT   },      // PDE
      };
      dispatch(ctx, table, i, rel, sym);
      break;
    }
    default:
      Error(ctx) << *this << ": unknown relocation: " << rel;
    }
  }
}

} // namespace mold::elf
