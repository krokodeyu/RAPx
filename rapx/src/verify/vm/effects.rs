//! Apply modeled call effects to symbolic state.

use rustc_hir::def_id::DefId;
use rustc_middle::mir::{Local, Operand};
use rustc_middle::ty::TyKind;
use z3::ast::{Ast, Bool, Int};

use super::state::{
    AllocId, AllocKind, ElementTy, OffsetKind, Provenance, ValueFacts, ValueSource, VmState,
    VmValue,
};
use crate::helpers::mir_utils;
use crate::verify::api_classify;
use crate::verify::call_summary::{BoundedArithmeticOp, CallEffect, OverflowOp};

impl<'z3, 'tcx> VmState<'z3, 'tcx> {
    pub(crate) fn slice_index_start(
        &self,
        index: &VmValue<'z3, 'tcx>,
        index_local: Local,
    ) -> Option<Int<'z3>> {
        let start_field = || {
            self.field_value(index_local, &[0])
                .map(|value| value.z3_term.clone())
        };
        match index.ty.kind() {
            TyKind::Uint(rustc_middle::ty::UintTy::Usize) => Some(index.z3_term.clone()),
            TyKind::Adt(def, _) => match mir_utils::range_kind(self.tcx, def.did()) {
                mir_utils::RangeKind::RangeTo => Some(Int::from_u64(self.z3_ctx, 0)),
                mir_utils::RangeKind::RangeFrom
                | mir_utils::RangeKind::Range
                | mir_utils::RangeKind::RangeInclusive => start_field(),
                mir_utils::RangeKind::Other => {
                    use crate::def_ids::lang_api::{self, LangItem};
                    if api_classify::is_index_range(def.did()) {
                        start_field()
                    } else {
                        (lang_api::matches(self.tcx, def.did(), LangItem::RangeToInclusive)
                            || lang_api::matches(self.tcx, def.did(), LangItem::RangeFull))
                        .then(|| Int::from_u64(self.z3_ctx, 0))
                    }
                }
            },
            _ => None,
        }
    }

    pub(crate) fn slice_index_output_len(
        &self,
        base: &VmValue<'z3, 'tcx>,
        index: &VmValue<'z3, 'tcx>,
        index_local: Local,
    ) -> Option<Int<'z3>> {
        let TyKind::Adt(def, _) = index.ty.kind() else {
            return None;
        };
        let start = self.slice_index_start(index, index_local)?;
        let one = Int::from_u64(self.z3_ctx, 1);
        let end = match mir_utils::range_kind(self.tcx, def.did()) {
            mir_utils::RangeKind::Range => self.field_value(index_local, &[1])?.z3_term.clone(),
            mir_utils::RangeKind::RangeInclusive => {
                self.field_value(index_local, &[1])?.z3_term.clone() + &one
            }
            mir_utils::RangeKind::RangeTo => self.field_value(index_local, &[0])?.z3_term.clone(),
            mir_utils::RangeKind::RangeFrom => self.slice_len_from_value(base)?,
            mir_utils::RangeKind::Other if api_classify::is_index_range(def.did()) => {
                self.field_value(index_local, &[1])?.z3_term.clone()
            }
            mir_utils::RangeKind::Other => {
                use crate::def_ids::lang_api::{self, LangItem};
                if lang_api::matches(self.tcx, def.did(), LangItem::RangeToInclusive) {
                    self.field_value(index_local, &[0])?.z3_term.clone() + &one
                } else if lang_api::matches(self.tcx, def.did(), LangItem::RangeFull) {
                    self.slice_len_from_value(base)?
                } else {
                    return None;
                }
            }
        };
        Some(end - &start)
    }

    pub(crate) fn apply_slice_offset(
        &mut self,
        dest: Local,
        base: &VmValue<'z3, 'tcx>,
        index: &VmValue<'z3, 'tcx>,
        index_local: Local,
    ) -> bool {
        let start = if let Some(start) = self.slice_index_start(index, index_local) {
            start
        } else {
            let Some(len) = self.slice_len_from_value(base) else {
                return false;
            };
            let start = self.fresh_int("slice_offset_index");
            let zero = Int::from_u64(self.z3_ctx, 0);
            self.constraints.assertions.push(start.ge(&zero));
            let dest_ty = self.body().local_decls[dest].ty;
            let is_slice = mir_utils::pointee_ty(dest_ty).is_some_and(|ty| {
                matches!(ty.kind(), TyKind::Slice(_) | TyKind::Str)
            });
            self.constraints.assertions.push(if is_slice {
                start.le(&len)
            } else {
                start.lt(&len)
            });
            start
        };
        let offset = VmValue::new(start, self.tcx.types.usize);
        self.apply_pointer_add(dest, base, &offset, None, base.facts.in_bounds);
        true
    }

    pub(crate) fn copy_memory_values(
        &mut self,
        source: &VmValue<'z3, 'tcx>,
        target: &VmValue<'z3, 'tcx>,
        count: &VmValue<'z3, 'tcx>,
    ) {
        let concrete_count = count.z3_term.simplify().as_u64();
        if concrete_count == Some(0) {
            return;
        }
        let Some(pointee_ty) = mir_utils::pointee_ty(target.ty) else {
            self.unsupported_mir = true;
            return;
        };
        let local_target = self
            .find_local_by_address(&target.z3_term)
            .is_some_and(|local| self.body().local_decls[local].ty == pointee_ty);
        let inline_target = self
            .inline
            .arg_referents
            .iter()
            .enumerate()
            .any(|(i, referent)| {
                referent.is_some()
                    && self
                        .local_value(Local::from_usize(i + 1))
                        .is_some_and(|value| value.z3_term == target.z3_term)
            });
        let slice_target = target.provenance.as_ref().is_some_and(|prov| {
            matches!(self.alloc(prov.alloc_id).kind, AllocKind::Slice { .. })
        });
        let allocated_target = target.provenance.as_ref().is_some_and(|prov| {
            let alloc = self.alloc(prov.alloc_id);
            matches!(&alloc.kind, AllocKind::Object | AllocKind::Slice { .. })
                && alloc.element_ty.as_ty() == Some(pointee_ty)
                && (matches!(&alloc.kind, AllocKind::Slice { .. })
                    || prov.offset.simplify().as_u64() == Some(0))
        });
        let scratch_target_ty = target.provenance.as_ref().and_then(|prov| {
            let alloc = self.alloc(prov.alloc_id);
            let ty = alloc.element_ty.as_ty()?;
            match ty.kind() {
                TyKind::Adt(def, args)
                    if api_classify::is_maybe_uninit_type(def.did())
                        && prov.offset.simplify().as_u64() == Some(0) => {
                    args.first().and_then(|arg| arg.as_type())
                }
                _ => None,
            }
        });
        let scratch_target = scratch_target_ty.is_some();
        if !local_target && !inline_target && !allocated_target && !scratch_target {
            self.unsupported_mir = true;
            return;
        }
        if concrete_count != Some(1) && target.provenance.is_none() {
            self.unsupported_mir = true;
            return;
        }
        let source_value = self
            .find_local_by_address(&source.z3_term)
            .filter(|local| self.body().local_decls[*local].ty == pointee_ty)
            .and_then(|local| self.local_value(local).cloned())
            .or_else(|| {
                source
                    .provenance
                    .as_ref()
                    .and_then(|prov| self.load_value(prov.alloc_id, pointee_ty, &[]).cloned())
            });
        let source_initialized = source
            .provenance
            .as_ref()
            .is_some_and(|prov| {
                let content = self.content(prov.alloc_id);
                let maybe_uninit = self.alloc(prov.alloc_id).element_ty.as_ty().is_some_and(|ty| {
                    matches!(ty.kind(), TyKind::Adt(def, _)
                        if api_classify::is_maybe_uninit_type(def.did()))
                });
                (content.facts.initialized && !maybe_uninit)
                    || (prov.offset.simplify().as_u64() == Some(0)
                        && content.copied_prefix.as_ref().is_some_and(|(ty, copied_count)| {
                            *ty == pointee_ty
                                && self.path_implies(&count.z3_term.le(copied_count))
                        }))
            })
            || source_value.as_ref().is_some_and(|value| {
                value.facts.init && (source.provenance.is_none() || concrete_count == Some(1))
            });
        if (slice_target || scratch_target) && !source_initialized {
            self.unsupported_mir = true;
            return;
        }
        let mut copied = if concrete_count == Some(1) {
            source_value.unwrap_or_else(|| VmValue::new(self.fresh_int("copy_value"), pointee_ty))
        } else {
            VmValue::new(self.fresh_int("copy_value"), pointee_ty)
        };
        copied.facts.init |= source_initialized;
        if scratch_target {
            let alloc_id = target.provenance.as_ref().unwrap().alloc_id;
            self.forget_memory_values(alloc_id);
            if concrete_count == Some(1) && scratch_target_ty == Some(pointee_ty) {
                self.store_value(alloc_id, pointee_ty, Vec::new(), copied);
                self.content_mut(alloc_id).facts.initialized = true;
                self.content_mut(alloc_id).facts.written_pointee_ty = Some(pointee_ty);
            } else {
                self.content_mut(alloc_id).copied_prefix =
                    Some((pointee_ty, count.z3_term.clone()));
            }
            return;
        }
        if let Some(prov) = &target.provenance {
            self.forget_memory_values(prov.alloc_id);
        }
        if concrete_count == Some(1) {
            self.write_through_pointer(target, Vec::new(), copied);
        }
    }

    fn forget_memory_values(&mut self, alloc_id: AllocId) {
        let root = self.root_alloc(alloc_id);
        self.clear_write_evidence(root);
        let aliases: Vec<_> = (0..self.units.len())
            .map(AllocId)
            .filter(|id| self.root_alloc(*id) == root)
            .collect();
        for id in aliases {
            self.alloc_mut(id).facts.for_each = Default::default();
            let content = self.content_mut(id);
            content.values.clear();
            content.indexed_elements.clear();
            content.indexed_elements_modified = true;
            content.copied_prefix = None;
            content.facts.written_pointee_ty = None;
            content.byte_array = None;
            content.byte_written.clear();
            content.facts.cstr_trusted = false;
            content.facts.utf8_trusted = false;
            content.facts.split_transmute_to = None;
        }
    }

    /// Abstract a safe `Fn*` call when only a scalar boolean or `Ordering`
    /// result is required. The closure body is unavailable at a generic call
    /// site, so every allocation outside the active frame may have been
    /// reached through captures or globals. Locals explicitly passed by
    /// reference are also invalidated. Frame-private counters remain stable.
    pub(crate) fn apply_callback_effect(
        &mut self,
        args: &[VmValue<'z3, 'tcx>],
        caller_arg_locals: &[Option<Local>],
        dest: Local,
        callee: DefId,
    ) {
        let ty = self.body().local_decls[dest].ty;
        if !crate::verify::call_summary::callback::supports_result(ty) {
            self.record_missing_mir(callee);
            return;
        }
        let input_receiver = caller_arg_locals
            .first()
            .copied()
            .flatten()
            .is_some_and(|local| local.as_usize() <= self.body().arg_count)
            || args
                .first()
                .and_then(VmValue::provenance_alloc_id)
                .is_some_and(|id| {
                    (1..=self.body().arg_count).any(|index| {
                        self.current_frame
                            .local_alloc
                            .get(&Local::from_usize(index))
                            == Some(&id)
                    })
                });
        if !input_receiver {
            self.record_missing_mir(callee);
            return;
        }
        // The rust-call tuple may carry references to this frame's locals.
        // Require provenance for every reference field before keeping those
        // locals private. An opaque aggregate can hide such a reference.
        for (index, arg) in args.iter().enumerate().skip(1) {
            let resolved = match arg.ty.kind() {
                TyKind::Ref(..) | TyKind::RawPtr(..) => arg.provenance.is_some(),
                TyKind::Tuple(fields) => fields.iter().enumerate().all(|(field, ty)| {
                    match ty.kind() {
                        TyKind::Ref(..) | TyKind::RawPtr(..) => caller_arg_locals
                            .get(index)
                            .copied()
                            .flatten()
                            .and_then(|local| self.current_frame.local_alloc.get(&local))
                            .and_then(|id| self.load_value(*id, arg.ty, &[field]))
                            .and_then(VmValue::provenance_alloc_id)
                            .is_some(),
                        TyKind::Bool | TyKind::Char | TyKind::Int(_) | TyKind::Uint(_)
                        | TyKind::Float(_) | TyKind::Never => true,
                        _ => false,
                    }
                }),
                _ => false,
            };
            if !resolved {
                self.record_missing_mir(callee);
                return;
            }
        }

        let private: Vec<_> = self.current_frame.local_alloc.values().copied().collect();
        let borrowed: Vec<_> = self
            .current_frame
            .local_alloc
            .iter()
            .filter_map(|(local, id)| {
                (matches!(self.body().local_decls[*local].ty.kind(), TyKind::Ref(..))
                    && !self.alloc(*id).facts.dead)
                    .then(|| self.local_value(*local))
                    .flatten()
                    .and_then(VmValue::provenance_alloc_id)
                    .map(|id| self.root_alloc(id))
            })
            .collect();
        let mut affected: Vec<_> = (0..self.units.len())
            .map(AllocId)
            .filter(|id| !private.contains(id))
            .collect();
        affected.extend(args.iter().filter_map(VmValue::provenance_alloc_id));
        for local in caller_arg_locals.iter().flatten() {
            if let Some(id) = self.current_frame.local_alloc.get(local) {
                affected.extend(
                    self.content(*id)
                        .values
                        .values()
                        .filter_map(VmValue::provenance_alloc_id),
                );
            }
        }
        affected = affected.into_iter().map(|id| self.root_alloc(id)).collect();
        affected.sort_by_key(|id| id.0);
        affected.dedup();
        for id in affected {
            if !borrowed.contains(&id) {
                self.callback_may_drop.insert(id);
            }
            self.forget_memory_values(id);
            for alias in 0..self.units.len() {
                let alias = AllocId(alias);
                if self.root_alloc(alias) == id {
                    let facts = &mut self.alloc_mut(alias).facts;
                    facts.owning = None;
                    if !borrowed.contains(&id) {
                        facts.liveness = None;
                    }
                }
            }
        }

        let term = self.fresh_int("callback_result");
        if matches!(ty.kind(), TyKind::Bool) {
            let zero = Int::from_u64(self.z3_ctx, 0);
            let one = Int::from_u64(self.z3_ctx, 1);
            self.constraints.assertions.push(term.ge(&zero));
            self.constraints.assertions.push(term.le(&one));
        }
        let mut result = VmValue::new(term, ty);
        result.facts.init = true;
        self.set_local(dest, result);
    }

    /// Apply a single call effect to the VM state.
    pub(crate) fn apply_call_effect(
        &mut self,
        effect: &CallEffect,
        args: &[VmValue<'z3, 'tcx>],
        caller_arg_locals: &[Option<Local>],
        dest: Local,
        callee: Option<DefId>,
        func: &Operand<'tcx>,
    ) {
        match effect {
            CallEffect::ReturnUninitWrapper => {
                self.ensure_local_allocation(dest);
                if let Some(&id) = self.current_frame.local_alloc.get(&dest) {
                    self.content_mut(id).facts.inner_uninit = true;
                }
            }
            CallEffect::ReturnLayoutSymbolic { is_size } => {
                let ty = mir_utils::fn_def_first_type_arg(func)
                    .expect("layout effect built from a FnDef");
                let term = if *is_size {
                    self.size_sym(ty)
                } else {
                    self.align_sym(ty)
                };
                let dest_ty = self.body().local_decls[dest].ty;
                self.set_local(dest, VmValue::new(term, dest_ty));
            }
            CallEffect::ReturnAliasArg { arg } => {
                if let Some(arg_val) = args.get(*arg) {
                    self.set_dest_as_heap_ptr(arg_val, dest);
                }
            }
            CallEffect::ReturnArgValue { arg } => {
                if let Some(arg_val) = args.get(*arg) {
                    let mut val = arg_val.clone();
                    val.ty = self.body().local_decls[dest].ty;
                    val.facts.init = true;
                    self.set_local(dest, val);
                }
            }
            CallEffect::ReturnOpaque => {
                self.set_opaque_intrinsic_result(dest);
            }
            CallEffect::ReturnSizeOfVal { arg } => {
                let size = args.get(*arg).and_then(|value| {
                    let pointee = mir_utils::pointee_ty(value.ty)?;
                    match pointee.kind() {
                        TyKind::Slice(elem) => {
                            let len = self.slice_len_from_value(value)?;
                            Some(Int::mul(self.z3_ctx, &[&len, &self.size_sym(*elem)]))
                        }
                        TyKind::Dynamic(..) => None,
                        _ => Some(self.size_sym(pointee)),
                    }
                });
                if let Some(size) = size {
                    let ty = self.body().local_decls[dest].ty;
                    self.set_local(dest, VmValue::new(size, ty));
                } else {
                    self.set_opaque_intrinsic_result(dest);
                }
            }
            CallEffect::ReturnRawEq {
                left_arg,
                right_arg,
            } => {
                let ty = self.body().local_decls[dest].ty;
                let Some((left, right)) = args.get(*left_arg).zip(args.get(*right_arg)) else {
                    self.unsupported_mir = true;
                    return;
                };
                if left.z3_term == right.z3_term && left.ty == right.ty {
                    self.set_local(dest, VmValue::new(Int::from_u64(self.z3_ctx, 1), ty));
                } else if let Some((left_local, right_local)) = self
                    .find_local_by_address(&left.z3_term)
                    .zip(self.find_local_by_address(&right.z3_term))
                    .filter(|(left_local, right_local)| {
                        let left_ty = self.body().local_decls[*left_local].ty;
                        left_ty == self.body().local_decls[*right_local].ty
                            && matches!(left_ty.kind(), TyKind::Int(_) | TyKind::Uint(_))
                    })
                    && let Some((left_value, right_value)) = self
                        .local_value(left_local)
                        .zip(self.local_value(right_local))
                {
                    let one = Int::from_u64(self.z3_ctx, 1);
                    let zero = Int::from_u64(self.z3_ctx, 0);
                    let term = left_value
                        .z3_term
                        ._eq(&right_value.z3_term)
                        .ite(&one, &zero);
                    self.set_local(dest, VmValue::new(term, ty));
                } else {
                    let term = self.fresh_int("raw_eq_result");
                    self.inline.opaque_intrinsic_terms.insert(term.clone());
                    self.set_local(dest, VmValue::new(term, ty));
                }
            }
            CallEffect::ReturnExactDiv { lhs_arg, rhs_arg } => {
                let ty = self.body().local_decls[dest].ty;
                let Some((lhs, rhs)) = args.get(*lhs_arg).zip(args.get(*rhs_arg)) else {
                    self.unsupported_mir = true;
                    return;
                };
                if !matches!(ty.kind(), TyKind::Uint(_)) {
                    self.unsupported_mir = true;
                    return;
                }
                self.set_local(dest, VmValue::new(lhs.z3_term.div(&rhs.z3_term), ty));
            }
            CallEffect::SelectUnpredictable => {
                if args.len() >= 3 {
                    let term = self.fresh_int(&format!("selunpred_{}", dest.as_usize()));
                    let dest_ty = self.body().local_decls[dest].ty;
                    let eq1 = term._eq(&args[1].z3_term);
                    let eq2 = term._eq(&args[2].z3_term);
                    self.constraints
                        .assertions
                        .push(Bool::or(self.z3_ctx, &[&eq1, &eq2]));
                    let prov = args[1]
                        .provenance
                        .clone()
                        .or_else(|| args[2].provenance.clone());
                    self.set_local(
                        dest,
                        VmValue {
                            z3_term: term,
                            ty: dest_ty,
                            provenance: prov,
                            facts: ValueFacts::default(),
                            source: ValueSource::None,
                        },
                    );
                }
            }
            CallEffect::ReturnDerefArg { arg } => {
                let dest_ty = self.body().local_decls[dest].ty;
                let mut val = args
                    .get(*arg)
                    .cloned()
                    .unwrap_or_else(|| VmValue::new(self.fresh_int("replaced"), dest_ty));
                // `mem::replace(&mut dest, src)` returns `*dest`: the pointee
                // value, not the `&mut` reference.  With M2 the reference's
                // `path == []` holds its *address* (whole value), so the pointee
                // is recovered from the materialized field maps by type +
                // provenance (e.g. `self.v: *mut [T]`), or — when the borrow
                // chain was dropped by the slicer and no field value is
                // recoverable — modeled as a fresh external allocation so a
                // downstream `Allocated`/`InBound` can still match `[T]` vs `T`.
                if let Some(search) = self
                    .units
                    .iter()
                    .flat_map(|u| u.content.values.values())
                    .find(|v| v.ty == dest_ty && v.is_pointer())
                    .cloned()
                {
                    val = search;
                } else if let Some(elem) = mir_utils::pointee_ty(dest_ty) {
                    let is_slice = matches!(elem.kind(), rustc_middle::ty::TyKind::Slice(_));
                    if is_slice {
                        let elem_align = self.align_sym(elem);
                        let (alloc_id, base) = self.allocate_external(
                            Int::from_u64(self.z3_ctx, i64::MAX as u64),
                            elem_align,
                            Some(elem),
                        );
                        val = VmValue {
                            z3_term: base,
                            ty: dest_ty,
                            provenance: Some(Provenance {
                                alloc_id,
                                offset: Int::from_u64(self.z3_ctx, 0),
                                offset_kind: None,
                            }),
                            facts: ValueFacts::default(),
                            source: ValueSource::None,
                        };
                    }
                }
                val.ty = dest_ty;
                self.set_local(dest, val);
            }
            CallEffect::ReturnTransparentDeref { arg, peel } => {
                if let Some(arg_val) = args.get(*arg) {
                    self.set_dest_as_heap_ptr(arg_val, dest);
                    if let Some(arg_local) = caller_arg_locals.get(*arg).copied().flatten() {
                        let keys: Vec<Vec<usize>> = self.field_paths(arg_local);
                        for path in keys {
                            if path.len() > *peel
                                && path[..*peel].iter().all(|&f| f == 0)
                                && let Some(v) = self.field_value(arg_local, &path).cloned()
                            {
                                self.set_field_value(dest, path[*peel..].to_vec(), v);
                            }
                        }
                    }
                }
            }
            CallEffect::ReturnTupleFieldLength {
                field: _field,
                from_arg: _from_arg,
            } => {
                if args.len() < 2 {
                    return;
                }
                let self_val = &args[0]; // &[T]
                let mid_val = &args[1]; // usize

                let dest_ty = self.body().local_decls[dest].ty;
                if let TyKind::Tuple(elem_tys) = dest_ty.kind() {
                    let src_alloc_id = self_val.provenance.as_ref().map(|p| p.alloc_id);

                    let (elem_ty, elem_sz_term, alloc_size) = src_alloc_id
                        .map(|id| self.alloc(id))
                        .map(|a| {
                            let ty = a.element_ty.as_ty();
                            let sz_term = self.size_sym_read(ty.unwrap_or(self_val.ty));
                            (ty, sz_term, a.size.clone())
                        })
                        .unwrap_or_else(|| {
                            // Provenance lost (e.g. `mem::replace` on a raw field
                            // whose borrow the slicer dropped): fall back to the
                            // slice pointee type so `InBound`/`Allocated` can
                            // still match `[T]` against the element `T`.
                            let pointee = mir_utils::pointee_ty(self_val.ty);
                            let sz = self.size_sym_read(pointee.unwrap_or(self_val.ty));
                            (pointee, sz, Int::from_u64(self.z3_ctx, 1))
                        });

                    let total_len = self
                        .slice_len_from_value(self_val)
                        .unwrap_or_else(|| alloc_size.div(&elem_sz_term)); // self.len()

                    let zero = Int::from_u64(self.z3_ctx, 0);
                    self.constraints.assertions.push(mid_val.z3_term.ge(&zero));
                    self.constraints
                        .assertions
                        .push(mid_val.z3_term.le(&total_len));

                    let mid = mid_val.z3_term.clone();
                    let rest_len = Int::sub(self.z3_ctx, &[&total_len, &mid]);

                    let mid_bytes = Int::mul(self.z3_ctx, &[&mid, &elem_sz_term]);
                    let ptr1 = Int::add(self.z3_ctx, &[&self_val.z3_term, &mid_bytes]);

                    for f in 0..elem_tys.len() {
                        let field_ty = elem_tys[f];
                        let (field_len, field_ptr) = if f == 0 {
                            (mid.clone(), self_val.z3_term.clone())
                        } else {
                            (rest_len.clone(), ptr1.clone())
                        };
                        let field_size = Int::mul(self.z3_ctx, &[&field_len, &elem_sz_term]);
                        let field_alloc_align = self_val
                            .provenance
                            .as_ref()
                            .map(|p| self.alloc(p.alloc_id).align.clone())
                            .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));

                        let (alloc_id, _base) = self.allocate_slice(
                            field_len.clone(),
                            elem_sz_term.clone(),
                            field_alloc_align.clone(),
                            elem_ty,
                        );
                        let src_bytes = Int::mul(self.z3_ctx, &[&total_len, &elem_sz_term]);
                        if f == 0 {
                            self.constraints.assertions.push(field_size._eq(&mid_bytes));
                        } else {
                            let remaining = Int::sub(self.z3_ctx, &[&src_bytes, &mid_bytes]);
                            self.constraints.assertions.push(field_size._eq(&remaining));
                        }
                        self.content_mut(alloc_id).facts.initialized = true;
                        if let Some(ref source_prov) = self_val.provenance {
                            self.alloc_mut(alloc_id).parent = Some(source_prov.alloc_id);
                        }

                        let field_offset = Int::from_u64(self.z3_ctx, 0);

                        let field_prov = Provenance {
                            alloc_id,
                            offset: field_offset,
                            offset_kind: None,
                        };

                        let field_val = VmValue {
                            z3_term: field_ptr,
                            ty: field_ty,
                            provenance: Some(field_prov),
                            facts: ValueFacts {
                                init: true,
                                non_null: true,
                                dangling: false,
                                in_bounds: true,
                                align_n: Some(field_alloc_align),
                            },
                            source: ValueSource::None,
                        };
                        self.set_field_value(dest, vec![f], field_val);
                    }
                }
            }
            CallEffect::ReturnRange { bounds_arg } => {
                self.apply_range_effect(*bounds_arg, args, caller_arg_locals, dest);
            }
            CallEffect::ReturnPointerFromArg { arg } => {
                if let Some(arg_val) = args.get(*arg) {
                    let mut val = arg_val.clone();
                    let dest_ty = self.body().local_decls[dest].ty;
                    val.ty = dest_ty;
                    let src_non_null = arg_val.facts.non_null
                        || matches!(arg_val.ty.kind(), rustc_middle::ty::TyKind::Ref(..))
                        || matches!(
                            arg_val.ty.kind(),
                            rustc_middle::ty::TyKind::Adt(adt, _)
                                if api_classify::is_std_nonnull(adt.did())
                        );
                    val.facts.non_null = src_non_null;
                    val.facts.align_n = arg_val.facts.align_n.clone();
                    if matches!(dest_ty.kind(), rustc_middle::ty::TyKind::RawPtr(..)) {
                        val.facts.init = true;
                    }
                    // For heap-backed containers (Vec/CString/String) and slice
                    // views: redirect as_ptr() from the struct/slice allocation
                    // to the heap data allocation.
                    if let Some(ref prov) = val.provenance
                        && let Some(data_alloc) = self.data_alloc_of(prov.alloc_id, arg_val.ty)
                    {
                        val.z3_term = self.allocation_base(data_alloc).clone();
                        val.provenance = Some(Provenance {
                            alloc_id: data_alloc,
                            offset: Int::from_u64(self.z3_ctx, 0),
                            offset_kind: None,
                        });
                        val.facts.align_n = Self::non_trivial_align(self.alloc(data_alloc).align.clone());
                    }
                    if src_non_null {
                        let zero = Int::from_u64(self.z3_ctx, 0);
                        self.constraints
                            .assertions
                            .push(val.z3_term._eq(&zero).not());
                    }
                    self.set_local(dest, val);
                }
            }
            CallEffect::ReturnWrappingPointerOffset {
                base_arg,
                offset_arg,
            } => {
                if let (Some(base), Some(offset)) = (args.get(*base_arg), args.get(*offset_arg)) {
                    let stride = mir_utils::pointee_ty(base.ty).and_then(|pointee| {
                        let (_, size) = mir_utils::type_layout(
                            self.tcx,
                            self.current_frame.current_def_id,
                            pointee,
                        )?;
                        if size == 0 && mir_utils::ty_has_type_param(pointee) {
                            None
                        } else {
                            Some(size)
                        }
                    });
                    self.apply_pointer_add(dest, base, offset, stride, false);
                }
            }
            CallEffect::ReturnPtrOffsetFrom { ptr_arg, base_arg } => {
                let dest_ty = self.body().local_decls[dest].ty;
                let ptr = args.get(*ptr_arg).cloned();
                let base = args.get(*base_arg).cloned();
                let count = match (ptr.as_ref(), base.as_ref()) {
                    (Some(p), Some(b)) => self
                        .iter_remaining_len_from_ptrs(b, p, None)
                        .unwrap_or_else(|| {
                            self.fresh_int(&format!("ptr_offset_from_{}", dest.as_usize()))
                        }),
                    _ => self.fresh_int(&format!("ptr_offset_from_{}", dest.as_usize())),
                };
                self.set_local(
                    dest,
                    VmValue {
                        z3_term: count,
                        ty: dest_ty,
                        provenance: None,
                        facts: ValueFacts::default(),
                        source: ValueSource::None,
                    },
                );
            }
            CallEffect::ReturnNonZero => {
                let zero = Int::from_u64(self.z3_ctx, 0);
                if let Some(mut existing) = self.local_value(dest).cloned() {
                    existing.facts.non_null = true;
                    // Record the non-zero fact as a path condition so that a
                    // downstream `ValidNum(result != 0)` obligation (e.g.
                    // `NonZero::new_unchecked` after a bit-preserving operation)
                    // discharges against it.
                    self.constraints
                        .assertions
                        .push(existing.z3_term._eq(&zero).not());
                    self.set_local(dest, existing);
                } else {
                    let dest_ty = self.body().local_decls[dest].ty;
                    let term = self.fresh_int(&format!("ret_nz_{}", dest.as_usize()));
                    self.constraints.assertions.push(term._eq(&zero).not());
                    self.set_local(
                        dest,
                        VmValue {
                            z3_term: term,
                            ty: dest_ty,
                            provenance: None,
                            facts: ValueFacts {
                                non_null: true,
                                ..Default::default()
                            },
                            source: ValueSource::None,
                        },
                    );
                }
            }
            CallEffect::ReturnAligned => {
                if let Some(mut existing) = self.local_value(dest).cloned() {
                    if existing.facts.align_n.is_none() {
                        let dest_ty = self.body().local_decls[dest].ty;
                        if let Some(pointee) = mir_utils::pointee_ty(dest_ty) {
                            let a = self.align_sym(pointee);
                            if a.simplify().as_u64() != Some(1) {
                                existing.facts.align_n = Some(a);
                            }
                        }
                    }
                    self.set_local(dest, existing);
                } else {
                    let dest_ty = self.body().local_decls[dest].ty;
                    let term = self.fresh_int(&format!("ret_align_{}", dest.as_usize()));
                    self.set_local(
                        dest,
                        VmValue {
                            z3_term: term,
                            ty: dest_ty,
                            provenance: None,
                            facts: ValueFacts {
                                ..Default::default()
                            },
                            source: ValueSource::None,
                        },
                    );
                }
            }
            CallEffect::ReturnConst { value } => {
                let dest_ty = self.body().local_decls[dest].ty;
                let term = Int::from_u64(self.z3_ctx, *value);
                let val = VmValue::new(term, dest_ty);
                self.set_local(dest, val);
            }
            CallEffect::ReturnAlignOffset { ptr_arg, align_arg } => {
                let dest_ty = self.body().local_decls[dest].ty;
                let offset = self.fresh_int(&format!("align_offset_{}", dest.as_usize()));
                if let (Some(ptr_val), Some(align_val)) = (args.get(*ptr_arg), args.get(*align_arg))
                {
                    // `ptr.align_offset(align)` returns the offset in *elements* of
                    // the pointee type (not bytes), so the aligned address is
                    // `ptr + offset * size_of::<pointee>()`.  Record that
                    // `(ptr + offset*elem) % align == 0` with `0 <= offset < align`
                    // so a downstream `*(ptr.add(offset) as *const U)` can
                    // discharge `Align`.
                    let elem = mir_utils::pointee_ty(ptr_val.ty)
                        .map(|pointee| self.size_sym(pointee))
                        .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));
                    let byte_off = Int::mul(self.z3_ctx, &[&offset, &elem]);
                    let ptr_plus_off = Int::add(self.z3_ctx, &[&ptr_val.z3_term, &byte_off]);
                    let zero = Int::from_u64(self.z3_ctx, 0);
                    self.constraints
                        .assertions
                        .push(ptr_plus_off.rem(&align_val.z3_term)._eq(&zero));
                    self.constraints.assertions.push(offset.ge(&zero));
                    self.constraints
                        .assertions
                        .push(offset.lt(&align_val.z3_term));
                }
                let val = VmValue::new(offset, dest_ty);
                self.set_local(dest, val);
            }
            CallEffect::ReturnMin { lhs_arg, rhs_arg } => {
                // Build the min as a first-class `ite(lhs <= rhs, lhs, rhs)`
                // term rather than a fresh variable plus disjunction facts.
                // A fresh variable breaks downstream alignment/bounds
                // reasoning: e.g. `ptr.align_offset(8)` guarantees
                // `(ptr + offset) % 8 == 0`, but `offset.min(len)` would
                // then become an unrelated symbol and the `Align`/`InBound`
                // checks on `*(ptr.add(offset) as *const usize)` could no
                // longer discharge.  With an `ite`, the path conditions
                // (`offset < 8`, `len >= 16`) let the solver reduce
                // `ite(offset <= len, offset, len)` back to `offset`.
                self.apply_binary_num(dest, args, *lhs_arg, *rhs_arg, |lhs, rhs| {
                    lhs.le(rhs).ite(lhs, rhs)
                });
            }
            CallEffect::ReturnMax { lhs_arg, rhs_arg } => {
                self.apply_binary_num(dest, args, *lhs_arg, *rhs_arg, |lhs, rhs| {
                    lhs.ge(rhs).ite(lhs, rhs)
                });
            }
            CallEffect::ReturnAdd { lhs_arg, rhs_arg } => {
                self.apply_binary_num(dest, args, *lhs_arg, *rhs_arg, |lhs, rhs| {
                    Int::add(self.z3_ctx, &[lhs, rhs])
                });
            }
            CallEffect::ReturnMul { lhs_arg, rhs_arg } => {
                self.apply_binary_num(dest, args, *lhs_arg, *rhs_arg, |lhs, rhs| {
                    Int::mul(self.z3_ctx, &[lhs, rhs])
                });
            }
            CallEffect::ReturnBoundedArithmetic {
                lhs_arg,
                rhs_arg,
                op,
            } => {
                if let (Some(lhs), Some(rhs)) = (args.get(*lhs_arg), args.get(*rhs_arg)) {
                    let dest_ty = self.body().local_decls[dest].ty;
                    let Some((min, max, modulus, signed)) = self
                        .integer_limits(dest_ty)
                        .or_else(|| self.integer_limits(lhs.ty))
                    else {
                        return;
                    };
                    let raw = match op {
                        BoundedArithmeticOp::SaturatingAdd => {
                            Int::add(self.z3_ctx, &[&lhs.z3_term, &rhs.z3_term])
                        }
                        BoundedArithmeticOp::SaturatingSub | BoundedArithmeticOp::WrappingSub => {
                            Int::sub(self.z3_ctx, &[&lhs.z3_term, &rhs.z3_term])
                        }
                    };
                    let term = match op {
                        BoundedArithmeticOp::SaturatingAdd | BoundedArithmeticOp::SaturatingSub => {
                            raw.lt(&min).ite(&min, &raw.gt(&max).ite(&max, &raw))
                        }
                        BoundedArithmeticOp::WrappingSub => {
                            let wrapped = raw.modulo(&modulus);
                            if signed {
                                wrapped
                                    .gt(&max)
                                    .ite(&Int::sub(self.z3_ctx, &[&wrapped, &modulus]), &wrapped)
                            } else {
                                wrapped
                            }
                        }
                    };
                    self.set_local(dest, VmValue::new(term, dest_ty));
                }
            }
            CallEffect::ReturnWithOverflow {
                lhs_arg,
                rhs_arg,
                op,
            } => {
                if let (Some(lhs), Some(rhs)) = (args.get(*lhs_arg), args.get(*rhs_arg)) {
                    let term = match op {
                        OverflowOp::Add => Int::add(self.z3_ctx, &[&lhs.z3_term, &rhs.z3_term]),
                        OverflowOp::Sub => Int::sub(self.z3_ctx, &[&lhs.z3_term, &rhs.z3_term]),
                        OverflowOp::Mul => Int::mul(self.z3_ctx, &[&lhs.z3_term, &rhs.z3_term]),
                    };
                    self.set_field_value(dest, vec![0], VmValue::new(term, lhs.ty));
                }
            }
            CallEffect::ReturnOptionSomeScanIndex { self_arg } => {
                // `Iterator::position`/`find` return `Option<usize>` whose `Some`
                // payload is a scan index into the iterator, so `0 <= i < self.len()`.
                // The receiver is `&mut iter` (a reference to the Iter/IterMut
                // struct), so resolve the reference to the iterator local it
                // points at (via its provenance = the iterator's stack alloc).
                // The iterator carries `ptr` (field 0) and `end_or_len`
                // (field 1); `len = end_or_len - ptr`.
                if let Some(iter_ref) = caller_arg_locals.get(*self_arg).copied().flatten() {
                    let iter_local = self
                        .local_value(iter_ref)
                        .and_then(|v| v.provenance_alloc_id())
                        .and_then(|alloc| {
                            self.current_frame
                                .local_alloc
                                .iter()
                                .find(|(_, a)| **a == alloc)
                                .map(|(l, _)| *l)
                        });
                    let ptr_term = iter_local
                        .and_then(|l| self.field_value(l, &[0]).map(|v| v.z3_term.clone()));
                    let end_term = iter_local
                        .and_then(|l| self.field_value(l, &[1]).map(|v| v.z3_term.clone()));
                    if let (Some(ptr), Some(end)) = (ptr_term, end_term) {
                        let len = Int::sub(self.z3_ctx, &[&end, &ptr]);
                        let payload = self.fresh_int(&format!("scan_idx_{}", dest.as_usize()));
                        self.constraints.assertions.push(payload.lt(&len));
                        let dest_ty = self.body().local_decls[dest].ty;
                        let payload_ty = match dest_ty.kind() {
                            TyKind::Adt(adt, substs) if adt.is_enum() => substs.type_at(0),
                            _ => dest_ty,
                        };
                        self.set_field_value(dest, vec![0], VmValue::new(payload, payload_ty));
                    }
                }
            }
            CallEffect::ReturnBranchPayload { arg } => {
                let dest_ty = self.body().local_decls[dest].ty;
                if let Some(value) = args.get(*arg)
                    && let Some(discr) = value.discriminant()
                    && let TyKind::Adt(adt, _) = value.ty.kind()
                {
                    let branch_discr = if self
                        .tcx
                        .is_diagnostic_item(rustc_span::sym::Option, adt.did())
                    {
                        let one = Int::from_u64(self.z3_ctx, 1);
                        Some(Int::sub(self.z3_ctx, &[&one, discr]))
                    } else if self
                        .tcx
                        .is_diagnostic_item(rustc_span::sym::Result, adt.did())
                    {
                        Some(discr.clone())
                    } else {
                        None
                    };
                    if let Some(branch_discr) = branch_discr {
                        let term = self.fresh_int("try_branch");
                        self.set_local(
                            dest,
                            VmValue {
                                z3_term: term,
                                ty: dest_ty,
                                provenance: None,
                                facts: ValueFacts::default(),
                                source: ValueSource::Discriminant(branch_discr),
                            },
                        );
                    }
                }
                let arg_local = caller_arg_locals.get(*arg).copied().flatten();
                if let Some(l) = arg_local
                    && let Some(payload) = self.field_value(l, &[0]).cloned()
                {
                    self.set_field_value(dest, vec![0], payload);
                }
            }
            CallEffect::ReturnNonZeroIff { arg } => {
                if let Some(a) = args.get(*arg) {
                    let dest_ty = self.body().local_decls[dest].ty;
                    let zero = Int::from_u64(self.z3_ctx, 0);
                    let term = self.fresh_int(&format!("ret_nz_iff_{}", dest.as_usize()));
                    self.constraints
                        .assertions
                        .push(term._eq(&zero)._eq(&a.z3_term._eq(&zero)));
                    self.set_local(dest, VmValue::new(term, dest_ty));
                }
            }
            CallEffect::ReturnOptionSomeNonZero => {
                let zero = Int::from_u64(self.z3_ctx, 0);
                let term = self.fresh_int(&format!("ret_opt_nz_{}", dest.as_usize()));
                self.constraints.assertions.push(term._eq(&zero).not());
                let payload_ty = args
                    .first()
                    .map(|a| a.ty)
                    .unwrap_or(self.body().local_decls[dest].ty);
                self.set_field_value(dest, vec![0], VmValue::new(term, payload_ty));
            }
            CallEffect::CopyMemory {
                source_arg,
                destination_arg,
                count_arg,
            } => {
                if let (Some(source), Some(target), Some(count)) = (
                    args.get(*source_arg),
                    args.get(*destination_arg),
                    args.get(*count_arg),
                ) {
                    self.copy_memory_values(source, target, count);
                }
            }
            CallEffect::SwapMemory {
                left_arg,
                right_arg,
            } => {
                let Some((left, right)) = args.get(*left_arg).zip(args.get(*right_arg)) else {
                    self.unsupported_mir = true;
                    return;
                };
                if !self.swap_memory_values(left, right) {
                    self.unsupported_mir = true;
                }
            }
            CallEffect::FillBytes {
                pointer_arg,
                byte_arg,
                count_arg,
            } => {
                let (Some(pointer), Some(byte), Some(count)) = (
                    args.get(*pointer_arg),
                    args.get(*byte_arg),
                    args.get(*count_arg),
                ) else {
                    self.unsupported_mir = true;
                    return;
                };
                let concrete_count = count.z3_term.simplify().as_u64();
                if concrete_count == Some(0) {
                    return;
                }
                let (Some(ty), Some(byte_value)) = (
                    mir_utils::pointee_ty(pointer.ty),
                    byte.z3_term.simplify().as_u64(),
                ) else {
                    self.unsupported_mir = true;
                    return;
                };
                let local = self
                    .find_local_by_address(&pointer.z3_term)
                    .filter(|local| self.body().local_decls[*local].ty == ty);
                let inline_target =
                    self.inline
                        .arg_referents
                        .iter()
                        .enumerate()
                        .any(|(i, referent)| {
                            referent.is_some()
                                && self
                                    .local_value(Local::from_usize(i + 1))
                                    .is_some_and(|value| value.z3_term == pointer.z3_term)
                        });
                if concrete_count != Some(1)
                    || (local.is_none() && !inline_target)
                    || !matches!(ty.kind(), TyKind::Int(_) | TyKind::Uint(_))
                    || byte_value > u8::MAX as u64
                {
                    self.unsupported_mir = true;
                    return;
                }
                let value_bits = if byte_value == 0 {
                    0
                } else {
                    let size = self.size_of_ty(ty);
                    if !matches!(ty.kind(), TyKind::Uint(_)) || size == 0 || size > 8 {
                        self.unsupported_mir = true;
                        return;
                    }
                    (0..size).fold(0_u64, |bits, index| bits | (byte_value << (8 * index)))
                };
                let mut value = VmValue::new(Int::from_u64(self.z3_ctx, value_bits), ty);
                value.facts.init = true;
                self.write_through_pointer(pointer, Vec::new(), value);
                if let Some(local) = local
                    && let Some(alloc_id) = self.current_frame.local_alloc.get(&local).copied()
                {
                    self.content_mut(alloc_id).facts.initialized = true;
                }
            }
            CallEffect::WriteMemory { pointer_arg } => {
                if let Some(arg_val) = args.get(*pointer_arg)
                    && let Some(prov) = &arg_val.provenance
                {
                    // Writing a non-`u8` value through a byte buffer reinterprets
                    // it (e.g. `*mut FreeBlock` cast from a `Vec<u8>` buffer):
                    // update the allocation's element type so a later `Typed`
                    // invariant matches the written type.
                    if let rustc_middle::ty::TyKind::RawPtr(inner, _)
                    | rustc_middle::ty::TyKind::Ref(_, inner, _) = arg_val.ty.kind()
                    {
                        let cur = self.alloc(prov.alloc_id).element_ty.as_ty();
                        let is_u8 = |t: rustc_middle::ty::Ty<'_>| {
                            matches!(
                                t.kind(),
                                rustc_middle::ty::TyKind::Uint(rustc_middle::ty::UintTy::U8)
                            )
                        };
                        if let Some(c) = cur
                            && is_u8(c)
                            && !is_u8(*inner)
                        {
                            self.alloc_mut(prov.alloc_id).element_ty = ElementTy::Typed(*inner);
                        }
                    }
                    let is_vec = api_classify::is_vec_push_or_reserve(callee);
                    let is_external = self.alloc(prov.alloc_id).is_external();
                    if is_vec && !is_external {
                        let elem_ty = match arg_val.ty.kind() {
                            TyKind::Ref(_, inner, _) | TyKind::RawPtr(inner, _) => {
                                crate::verify::call_summary::effects::vec_elem_ty(*inner)
                            }
                            _ => crate::verify::call_summary::effects::vec_elem_ty(arg_val.ty),
                        };
                        let heap_align = elem_ty
                            .map(|ty| self.align_sym(ty))
                            .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));
                        if let Some(old_data) = self.container_data_alloc(prov.alloc_id, arg_val.ty)
                        {
                            self.alloc_mut(old_data).facts.dead = true;
                        }
                        let max_size = Int::from_u64(self.z3_ctx, i64::MAX as u64);
                        let (data_alloc, data_base) =
                            self.allocate_external(max_size, heap_align, elem_ty);
                        let container_ty = match arg_val.ty.kind() {
                            TyKind::Ref(_, inner, _) | TyKind::RawPtr(inner, _) => *inner,
                            _ => arg_val.ty,
                        };
                        self.set_container_data_field(
                            prov.alloc_id,
                            arg_val.ty,
                            data_alloc,
                            data_base,
                            elem_ty.unwrap_or(container_ty),
                        );
                    }
                    let off_u64 = prov
                        .offset
                        .as_u64()
                        .or_else(|| prov.offset.simplify().as_u64());
                    let written_pointee_ty = self.alloc(prov.alloc_id).element_ty.as_ty().and_then(|ty| {
                        match ty.kind() {
                            TyKind::Adt(def, args) if api_classify::is_maybe_uninit_type(def.did()) => {
                                args.first().and_then(|arg| arg.as_type())
                                    .filter(|inner| Some(*inner) == mir_utils::pointee_ty(arg_val.ty))
                            }
                            _ => None,
                        }
                    });
                    if let Some(off) = off_u64 {
                        if off == 0 {
                            self.content_mut(prov.alloc_id).facts.initialized = true;
                        }
                        let elem_size = match arg_val.ty.kind() {
                            rustc_middle::ty::TyKind::Ref(_, inner, _) => {
                                self.size_of_ty(*inner) as usize
                            }
                            _ => 0,
                        };
                        let write_size = if elem_size > 0 {
                            elem_size
                        } else {
                            self.allocation_size(prov.alloc_id).as_u64().unwrap_or(0) as usize
                        };
                        let end = (off as usize + write_size).min(4096);
                        for byte_off in (off as usize)..end {
                            self.mark_byte_init(prov.alloc_id, byte_off);
                        }
                    } else {
                        // Symbolic write offset: the exact written element
                        // can't be tracked per-byte. For concrete allocation
                        // sizes, mark every byte (as before). For unknown /
                        // zero sizes — generic element types such as
                        // `MaybeUninit<T>` inside `[MaybeUninit<T>; N]` —
                        // mark the whole allocation initialized so a later
                        // `assume_init_read`/`assume_init_drop` can discharge
                        // `Init` on those (fully initialized) elements.
                        let size_val = self.allocation_size(prov.alloc_id).as_u64();
                        match size_val {
                            Some(sz) if sz > 0 => {
                                for off in 0..(sz as usize).min(1024) {
                                    self.mark_byte_init(prov.alloc_id, off);
                                }
                            }
                            _ => {
                                self.content_mut(prov.alloc_id).facts.initialized = true;
                            }
                        }
                    }
                    if off_u64 == Some(0) {
                        self.content_mut(prov.alloc_id).facts.written_pointee_ty = written_pointee_ty;
                    }
                }
            }
            CallEffect::ReturnFreshAllocation {
                pointer_arg,
                size_arg,
                elem_size,
            } => {
                if let (Some(ptr_val), Some(size_val)) =
                    (args.get(*pointer_arg), args.get(*size_arg))
                {
                    let dest_ty = self.body().local_decls[dest].ty;
                    let elem_ty = crate::verify::call_summary::effects::from_raw_parts_elem_ty(
                        self.tcx,
                        self.current_frame.current_def_id,
                        Some(dest),
                    );
                    // A generic element type uses the shared symbolic `sizeof_T`
                    // so the fresh allocation's size stays consistent with ptr
                    // strides and `InBound` cancels the factor.
                    let elem_sz_term = if *elem_size == 0 {
                        self.size_sym(elem_ty.unwrap_or(dest_ty))
                    } else {
                        Int::from_u64(self.z3_ctx, *elem_size)
                    };
                    let heap_align = elem_ty
                        .map(|ty| self.align_sym(ty))
                        .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));
                    let (alloc_id, base) = self.allocate_slice(
                        size_val.z3_term.clone(),
                        elem_sz_term.clone(),
                        heap_align,
                        elem_ty,
                    );
                    let prov = Provenance {
                        alloc_id,
                        offset: Int::from_u64(self.z3_ctx, 0),
                        offset_kind: None,
                    };
                    if let Some(ref source_prov) = ptr_val.provenance {
                        if !self.alloc(source_prov.alloc_id).facts.dead {
                            self.content_mut(alloc_id).facts.initialized = true;
                            self.alloc_mut(alloc_id).parent = Some(source_prov.alloc_id);
                        }
                        let src_offset = source_prov
                            .offset
                            .simplify()
                            .as_u64()
                            .map(|v| v as usize)
                            .unwrap_or(0);
                        self.copy_byte_tracking(source_prov.alloc_id, src_offset, alloc_id);
                    }
                    let result_align_n = ptr_val.facts.align_n.clone().or_else(|| {
                        ptr_val
                            .provenance
                            .as_ref()
                            .map(|p| self.alloc(p.alloc_id).align.clone())
                    });
                    let vec_base = base.clone();
                    let vec_prov = prov.clone();
                    let vec_len = size_val.z3_term.clone();
                    self.set_local(
                        dest,
                        VmValue {
                            z3_term: base,
                            ty: dest_ty,
                            provenance: Some(prov),
                            facts: ValueFacts {
                                non_null: true,
                                init: true,
                                in_bounds: true,
                                align_n: result_align_n.clone(),
                                dangling: false,
                            },
                            source: ValueSource::None,
                        },
                    );
                    if let rustc_middle::ty::TyKind::Adt(adt_def, _) = dest_ty.kind()
                        && api_classify::is_std_vec(adt_def.did())
                    {
                        let ptr_field = VmValue {
                            z3_term: vec_base,
                            ty: ptr_val.ty,
                            provenance: Some(vec_prov),
                            facts: ValueFacts {
                                non_null: true,
                                init: true,
                                in_bounds: true,
                                align_n: result_align_n,
                                dangling: false,
                            },
                            source: ValueSource::None,
                        };
                        self.materialize_vec_fields(dest, ptr_field, vec_len.clone(), vec_len);
                    }
                }
            }
            CallEffect::ReturnBoxAllocation => {
                let dest_ty = self.body().local_decls[dest].ty;
                // `pointee_ty` doesn't unwrap `Box`; extract its `T` from the
                // first generic argument so the heap allocation is sized to the
                // pointee and (below) the inner `Unique<T>.pointer` field can be
                // exposed.
                let pointee = mir_utils::pointee_ty(dest_ty).or_else(|| {
                    if let rustc_middle::ty::TyKind::Adt(adt, substs) = dest_ty.kind() {
                        if api_classify::is_std_box(adt.did()) {
                            substs.first().and_then(|s| s.as_type())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });
                let size = pointee
                    .map(|ty| self.size_sym(ty))
                    .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));
                let align = pointee
                    .map(|ty| self.align_sym(ty))
                    .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));
                let (alloc_id, base) = self.allocate(size, align, pointee);
                self.content_mut(alloc_id).facts.initialized = true;
                let align_n = pointee.map(|ty| self.align_sym(ty));
                let heap_prov = Provenance {
                    alloc_id,
                    offset: Int::from_u64(self.z3_ctx, 0),
                    offset_kind: None,
                };
                // Expose `Box`'s inner `Unique<T>.pointer` (`NonNull<T>` at path
                // `[0, 0]`) so inlined `Box::as_ptr`/`as_mut_ptr` bodies — which
                // read `(_1.0).0` and cast it to `*const`/`*mut T` — inherit the
                // heap pointer's provenance (rustc 1.95 lowers `&raw **b` to
                // exactly this field read + transmute).  Record it both on the
                // local's stack slot (`set_field_value`, for direct `_1.0.0`
                // reads) and on the heap allocation (`MemoryContent::values`, for
                // `(*&box).0.0` deref-reads through a reborrow).
                let nn_field = VmValue {
                    z3_term: base.clone(),
                    ty: dest_ty,
                    provenance: Some(heap_prov.clone()),
                    facts: ValueFacts {
                        non_null: true,
                        init: true,
                        ..Default::default()
                    },
                    source: ValueSource::None,
                };
                let nn_path = self
                    .container_ptr_field(dest_ty)
                    .expect("Box has no owning pointer field");
                self.set_field_value(dest, nn_path.clone(), nn_field.clone());
                self.units[alloc_id.0]
                    .content
                    .values
                    .insert((dest_ty, nn_path), nn_field);
                self.set_local(
                    dest,
                    VmValue {
                        z3_term: base.clone(),
                        ty: dest_ty,
                        provenance: Some(Provenance {
                            alloc_id,
                            offset: Int::from_u64(self.z3_ctx, 0),
                            offset_kind: None,
                        }),
                        facts: ValueFacts {
                            non_null: true,
                            init: true,
                            in_bounds: true,
                            align_n,
                            dangling: false,
                        },
                        source: ValueSource::None,
                    },
                );
            }
            CallEffect::ReturnExchangeMalloc { size_arg } => {
                if let Some(size_val) = args.get(*size_arg) {
                    let dest_ty = self.body().local_decls[dest].ty;
                    let u8_ty = self.tcx.types.u8;
                    let (alloc_id, base) = self.allocate_external(
                        size_val.z3_term.clone(),
                        Int::from_u64(self.z3_ctx, 1),
                        Some(u8_ty),
                    );
                    self.alloc_mut(alloc_id)
                        .set_slice_len(size_val.z3_term.clone());
                    self.content_mut(alloc_id).facts.initialized = true;
                    self.set_local(
                        dest,
                        VmValue {
                            z3_term: base,
                            ty: dest_ty,
                            provenance: Some(Provenance {
                                alloc_id,
                                offset: Int::from_u64(self.z3_ctx, 0),
                                offset_kind: None,
                            }),
                            facts: ValueFacts {
                                non_null: true,
                                init: true,
                                in_bounds: true,
                                ..ValueFacts::default()
                            },
                            source: ValueSource::None,
                        },
                    );
                }
            }
            CallEffect::ReturnNewAllocation {
                size_arg,
                elem_size,
            } => {
                if let Some(count) = args.get(*size_arg) {
                    self.apply_vec_allocation(dest, count, *elem_size, true);
                }
            }
            CallEffect::ReturnNewAllocationFromCap { cap_arg, elem_size } => {
                if let Some(cap) = args.get(*cap_arg) {
                    self.apply_vec_allocation(dest, cap, *elem_size, false);
                }
            }
            CallEffect::ReturnNewAllocationFromBox => {
                self.ensure_local_allocation(dest);
                let dest_ty = self.body().local_decls[dest].ty;
                let elem_ty = crate::verify::call_summary::effects::vec_elem_ty(dest_ty);
                let heap_align = elem_ty
                    .map(|ty| self.align_sym(ty))
                    .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));
                let known_len = args
                    .first()
                    .and_then(|v| v.provenance.as_ref())
                    .and_then(|p| {
                        let alloc = self.alloc(p.alloc_id);
                        if let Some(slice_len) = alloc.slice_len().cloned() {
                            return Some(slice_len);
                        }
                        // A boxed array (`box [T; N]`) has a concrete size but no
                        // slice length; its element count is `size / elem_size`.
                        let elem_size = elem_ty.map(|t| self.size_of_ty(t)).unwrap_or(1);
                        let n = alloc.size.as_u64()?;
                        Some(Int::from_u64(self.z3_ctx, n.checked_div(elem_size)?))
                    });
                let size = known_len
                    .clone()
                    .unwrap_or_else(|| Int::from_u64(self.z3_ctx, i64::MAX as u64));
                let (alloc_id, base) = self.allocate_external(size, heap_align, elem_ty);
                if let Some(box_alloc) = args.first().and_then(|v| v.provenance_alloc_id()) {
                    self.copy_byte_tracking(box_alloc, 0, alloc_id);
                }
                let dest_alloc_id = self.current_frame.local_alloc.get(&dest).copied();
                self.content_mut(alloc_id).facts.initialized = true;
                let vec_base = base.clone();
                self.set_local(
                    dest,
                    VmValue {
                        z3_term: base,
                        ty: dest_ty,
                        provenance: dest_alloc_id.map(|stack_id| Provenance {
                            alloc_id: stack_id,
                            offset: Int::from_u64(self.z3_ctx, 0),
                            offset_kind: None,
                        }),
                        facts: ValueFacts {
                            non_null: true,
                            init: true,
                            in_bounds: true,
                            ..ValueFacts::default()
                        },
                        source: ValueSource::None,
                    },
                );
                if let rustc_middle::ty::TyKind::Adt(adt_def, _) = dest_ty.kind()
                    && api_classify::is_std_vec(adt_def.did())
                {
                    let ptr_field = VmValue {
                        z3_term: vec_base,
                        ty: elem_ty.unwrap_or(dest_ty),
                        provenance: Some(Provenance {
                            alloc_id,
                            offset: Int::from_u64(self.z3_ctx, 0),
                            offset_kind: None,
                        }),
                        facts: ValueFacts {
                            non_null: true,
                            init: true,
                            in_bounds: true,
                            ..ValueFacts::default()
                        },
                        source: ValueSource::None,
                    };
                    let len_term = known_len
                        .clone()
                        .unwrap_or_else(|| self.fresh_int(&format!("vec_len_{}", dest.as_usize())));
                    self.materialize_vec_fields(dest, ptr_field, len_term.clone(), len_term);
                }
            }
            CallEffect::ReturnBoxFromVec { arg } => {
                if let Some(vec_val) = args.get(*arg)
                    && let Some(ref prov) = vec_val.provenance
                    && let Some(heap_alloc_id) =
                        self.container_data_alloc(prov.alloc_id, vec_val.ty)
                {
                    let heap_base = self.allocation_base(heap_alloc_id).clone();
                    let dest_ty = self.body().local_decls[dest].ty;
                    self.set_local(
                        dest,
                        VmValue {
                            z3_term: heap_base,
                            ty: dest_ty,
                            provenance: Some(Provenance {
                                alloc_id: heap_alloc_id,
                                offset: Int::from_u64(self.z3_ctx, 0),
                                offset_kind: None,
                            }),
                            facts: ValueFacts {
                                non_null: true,
                                init: true,
                                in_bounds: true,
                                ..ValueFacts::default()
                            },
                            source: ValueSource::None,
                        },
                    );
                }
            }
            CallEffect::OwnsInitMemory { arg } => {
                if let Some(arg_val) = args.get(*arg) {
                    if let Some(prov) = &arg_val.provenance {
                        self.content_mut(prov.alloc_id).facts.initialized = true;
                    }
                    let mut val = arg_val.clone();
                    val.ty = self.body().local_decls[dest].ty;
                    val.facts.init = true;
                    val.facts.non_null = true;
                    // `Box::from_raw`/`from_raw_in` reconstruct a Box whose
                    // `Unique<T>.pointer` (`NonNull<T>` at path `[0, 0]`) must
                    // carry the same provenance: rustc 1.95 lowers `Box::as_ptr`
                    // (`&raw **b`) to a `(_1.0).0` field read + transmute, so
                    // without this the re-derived pointer loses provenance.
                    if let rustc_middle::ty::TyKind::Adt(adt, _) = val.ty.kind()
                        && api_classify::is_std_box(adt.did())
                    {
                        let nn_path = self
                            .container_ptr_field(val.ty)
                            .expect("Box has no owning pointer field");
                        self.set_field_value(dest, nn_path.clone(), val.clone());
                        if let Some(prov) = &val.provenance {
                            self.units[prov.alloc_id.0]
                                .content
                                .values
                                .insert((val.ty, nn_path), val.clone());
                        }
                    }
                    self.set_local(dest, val);
                }
            }
            CallEffect::DropMemory { pointer_arg } => {
                // `ManuallyDrop::drop(slot)` / `drop_in_place(x)` frees the heap
                // allocation behind the argument. A reference/raw-pointer
                // argument carries the *stack* provenance of the referent
                // (penetrate to its heap field); a value argument (`Box`/`Vec`)
                // carries the heap provenance directly. Mark the allocation
                // dead; a second drop of an already-dead allocation is detected
                // downstream via `dead` alone.
                if let Some(arg_val) = args.get(*pointer_arg) {
                    let alloc_id = if matches!(
                        arg_val.ty.kind(),
                        rustc_middle::ty::TyKind::Ref(..) | rustc_middle::ty::TyKind::RawPtr(..)
                    ) {
                        self.find_local_by_address(&arg_val.z3_term)
                            .and_then(|r| self.owner_ptr_field(r))
                            .and_then(|v| v.provenance_alloc_id())
                    } else {
                        arg_val.provenance_alloc_id()
                    };
                    if let Some(alloc_id) = alloc_id {
                        self.alloc_mut(alloc_id).facts.dead = true;
                    }
                }
            }
            CallEffect::ReturnPowerOfTwo => {
                // `Layout::align()` returns the layout's alignment, which is a
                // non-zero power of two. `Layout::align` inlines to
                // `self.align.as_usize()`, whose transmute-based body drops the
                // `NonZero` provenance; re-establish the non-zero fact (and the
                // power-of-two fact) with a fresh symbol so downstream
                // `from_size_align_unchecked` can discharge `align != 0` (its
                // `(align & (align - 1)) == 0` check is otherwise vacuously
                // proved, since contract-level `BitAnd` is unsupported).
                let dest_ty = self.body().local_decls[dest].ty;
                let term = self.fresh_int(&format!("layout_align_{}", dest.as_usize()));
                let zero = Int::from_u64(self.z3_ctx, 0);
                self.constraints.assertions.push(term.gt(&zero));
                self.set_local(dest, VmValue::new(term, dest_ty));
            }
            CallEffect::ReturnSliceRangeIndex => {
                if args.len() < 2 {
                    return;
                }
                let dest_ty = self.body().local_decls[dest].ty;
                let is_slice = matches!(dest_ty.kind(), TyKind::Ref(_, inner, _)
                    if matches!(inner.kind(), TyKind::Slice(_)));
                let range_kind = args.get(1).and_then(|v| match v.ty.kind() {
                    TyKind::Adt(adt_def, _) => Some(mir_utils::range_kind(self.tcx, adt_def.did())),
                    _ => None,
                });
                if !is_slice && range_kind.is_none() {
                    return;
                }
                let Some(prov) = args[0].provenance.clone() else {
                    return;
                };
                let array_term = args[0].z3_term.clone();
                let elem_ty = match args[0].ty.kind() {
                    TyKind::Ref(_, inner, _) => match inner.kind() {
                        TyKind::Array(e, _) | TyKind::Slice(e) => *e,
                        _ => args[0].ty,
                    },
                    _ => args[0].ty,
                };
                let elem_align = self.align_sym(elem_ty);
                let elem_size = self.size_sym(elem_ty);
                let range_local = caller_arg_locals.get(1).copied().flatten();
                let range_field = |idx: usize| -> Option<Int<'z3>> {
                    range_local.and_then(|l| self.field_value(l, &[idx]).map(|v| v.z3_term.clone()))
                };
                let zero = Int::from_u64(self.z3_ctx, 0);
                let one = Int::from_u64(self.z3_ctx, 1);
                let total_len = self.slice_len_of_alloc(prov.alloc_id).unwrap_or_else(|| {
                    self.alloc(prov.alloc_id).size.div(&elem_size)
                });
                let (start, len) = match range_kind {
                    Some(mir_utils::RangeKind::RangeTo) => (
                        zero.clone(),
                        range_field(0).unwrap_or_else(|| total_len.clone()),
                    ),
                    Some(mir_utils::RangeKind::RangeFrom) => {
                        let s = range_field(0).unwrap_or_else(|| zero.clone());
                        (s.clone(), Int::sub(self.z3_ctx, &[&total_len, &s]))
                    }
                    Some(mir_utils::RangeKind::Range) => {
                        let s = range_field(0).unwrap_or_else(|| zero.clone());
                        let e = range_field(1).unwrap_or_else(|| total_len.clone());
                        (s.clone(), Int::sub(self.z3_ctx, &[&e, &s]))
                    }
                    Some(mir_utils::RangeKind::RangeInclusive) => {
                        let s = range_field(0).unwrap_or_else(|| zero.clone());
                        let e = range_field(1).unwrap_or_else(|| total_len.clone());
                        let l = Int::sub(self.z3_ctx, &[&e, &s]);
                        (s.clone(), Int::add(self.z3_ctx, &[&l, &one]))
                    }
                    _ => (zero.clone(), total_len.clone()),
                };
                let start_bytes = Int::mul(self.z3_ctx, &[&start, &elem_size]);
                let dest_term = Int::add(self.z3_ctx, &[&array_term, &start_bytes]);
                let (alloc_id, _) = self.allocate_slice(len, elem_size, elem_align, Some(elem_ty));
                self.alloc_mut(alloc_id).parent = Some(prov.alloc_id);
                self.set_local(
                    dest,
                    VmValue {
                        z3_term: dest_term,
                        ty: dest_ty,
                        provenance: Some(Provenance {
                            alloc_id,
                            offset: Int::from_u64(self.z3_ctx, 0),
                            offset_kind: None,
                        }),
                        facts: ValueFacts {
                            non_null: true,
                            init: true,
                            in_bounds: true,
                            ..Default::default()
                        },
                        source: ValueSource::None,
                    },
                );
            }
            CallEffect::ReturnIterLen { is_len } => {
                if args.is_empty() {
                    return;
                }
                let Some(local) = caller_arg_locals.first().copied().flatten() else {
                    return;
                };
                let Some((ptr, end)) = self.iter_ptr_end(local) else {
                    return;
                };
                let dest_ty = self.body().local_decls[dest].ty;
                if *is_len {
                    let len = self
                        .iter_len_from_ptrs(&ptr, &end)
                        .expect("iter_ptr_end guarantees same-alloc provenance");
                    self.set_local(dest, VmValue::new(len, dest_ty));
                } else {
                    let pp = ptr.provenance.as_ref().unwrap();
                    let ep = end.provenance.as_ref().unwrap();
                    let eq = pp.offset._eq(&ep.offset);
                    let zero = Int::from_u64(self.z3_ctx, 0);
                    let one = Int::from_u64(self.z3_ctx, 1);
                    let val = VmValue {
                        z3_term: eq.ite(&one, &zero),
                        ty: dest_ty,
                        provenance: None,
                        facts: ValueFacts::default(),
                        source: ValueSource::None,
                    };
                    self.set_local(dest, val);
                }
            }
            CallEffect::ReturnNonNullNew => {
                let Some(ptr) = args.first() else {
                    return;
                };
                let dest_ty = self.body().local_decls[dest].ty;
                let definitely_non_null = ptr.facts.non_null
                    || ptr.facts.in_bounds
                    || ptr
                        .provenance
                        .as_ref()
                        .is_some_and(|p| !self.alloc(p.alloc_id).is_external());
                if definitely_non_null {
                    let mut val = ptr.clone();
                    val.ty = dest_ty;
                    val.facts.non_null = true;
                    let zero = Int::from_u64(self.z3_ctx, 0);
                    self.constraints
                        .assertions
                        .push(ptr.z3_term._eq(&zero).not());
                    self.set_local(dest, val);
                } else {
                    let term = self.fresh_int(&format!("nn_new_{}", dest.as_usize()));
                    self.set_local(dest, VmValue::new(term, dest_ty));
                }
            }
            CallEffect::ReturnIterNext => {
                if args.is_empty() {
                    return;
                }
                let self_val = &args[0];
                let Some(local) = self.find_iter_self_local(self_val) else {
                    return;
                };
                let Some(iter_prov) = self_val.provenance.as_ref() else {
                    return;
                };
                let iter_key = (iter_prov.alloc_id, iter_prov.offset.simplify());
                let Some((ptr, end)) = self.iter_ptr_end(local) else {
                    return;
                };
                let pp = ptr.provenance.as_ref().unwrap();
                let ep = end.provenance.as_ref().unwrap();
                let buffer = ep.alloc_id;
                let dest_ty = self.body().local_decls[dest].ty;
                let sz = self.iter_elem_size(&ptr);
                let ep_elem = match &ep.offset_kind {
                    Some(OffsetKind::Element { index, .. }) => Some(index.clone()),
                    _ => None,
                };
                let ep_offset = ep.offset.clone();
                let remaining = self
                    .iter_remaining_len_from_ptrs(&ptr, &end, Some(&iter_key))
                    .expect("iter_ptr_end guarantees same-alloc provenance");
                let is_empty = remaining._eq(&Int::from_u64(self.z3_ctx, 0));
                let zero = Int::from_u64(self.z3_ctx, 0);
                let cur_off = match self.constraints.term_caches.iter_ptr_offset.get(&iter_key) {
                    Some((_, prev, _)) => Int::mul(self.z3_ctx, &[prev, &sz]),
                    None => pp.offset.clone(),
                };
                let old_ptr_val = VmValue {
                    z3_term: Int::add(self.z3_ctx, &[self.allocation_base(pp.alloc_id), &cur_off]),
                    ty: ptr.ty,
                    provenance: Some(Provenance {
                        alloc_id: pp.alloc_id,
                        offset: cur_off.clone(),
                        offset_kind: None,
                    }),
                    facts: ValueFacts {
                        non_null: true,
                        init: true,
                        ..Default::default()
                    },
                    source: ValueSource::None,
                };
                let one_term = Int::from_u64(self.z3_ctx, 1);
                let (old_offset, base_len_elem) =
                    match self.constraints.term_caches.iter_ptr_offset.get(&iter_key) {
                        Some((_, prev, base)) => (prev.clone(), base.clone()),
                        None => (pp.offset.div(&sz), ep_elem),
                    };
                let new_offset = Int::add(self.z3_ctx, &[&old_offset, &one_term]);
                let base_len = base_len_elem.clone().unwrap_or_else(|| ep_offset.div(&sz));
                self.constraints.assertions.push(Bool::or(
                    self.z3_ctx,
                    &[&is_empty, &new_offset.le(&base_len)],
                ));
                self.constraints.term_caches.iter_ptr_offset.insert(
                    iter_key,
                    (
                        buffer,
                        is_empty.ite(&old_offset, &new_offset),
                        base_len_elem,
                    ),
                );
                let result_val = VmValue {
                    z3_term: is_empty.ite(&zero, &old_ptr_val.z3_term),
                    ty: dest_ty,
                    provenance: None,
                    facts: ValueFacts::default(),
                    // Tie the Option's discriminant to the emptiness condition so
                    // `switchInt(discriminant(_n))` only takes the `Some` branch when
                    // the iterator was non-empty (and the `None` branch when empty).
                    source: ValueSource::Discriminant(is_empty.ite(&zero, &one_term)),
                };
                self.set_local(dest, result_val);
                self.set_field_value(dest, vec![0], old_ptr_val);
                let next_off = is_empty.ite(&cur_off, &Int::add(self.z3_ctx, &[&cur_off, &sz]));
                let mut next_ptr = ptr.clone();
                next_ptr.z3_term =
                    Int::add(self.z3_ctx, &[self.allocation_base(pp.alloc_id), &next_off]);
                next_ptr.provenance = Some(Provenance {
                    alloc_id: pp.alloc_id,
                    offset: next_off,
                    offset_kind: Some(OffsetKind::Element {
                        index: is_empty.ite(&old_offset, &new_offset),
                        stride: sz.clone(),
                    }),
                });
                let ptr_path = if self
                    .field_value(local, &[0])
                    .is_some_and(|field| field.z3_term == ptr.z3_term)
                {
                    vec![0]
                } else {
                    vec![0, 0]
                };
                self.set_field_value(local, ptr_path, next_ptr);
            }
            CallEffect::ReturnRangeNext => {
                if args.is_empty() {
                    return;
                }
                let self_val = &args[0];
                let TyKind::Ref(_, pointee, _) = self_val.ty.kind() else {
                    return;
                };
                let TyKind::Adt(adt_def, _) = pointee.kind() else {
                    return;
                };
                let Some(alloc_id) = self_val.provenance_alloc_id() else {
                    return;
                };
                // Field values use the monomorphized type, e.g. `Range<usize>`.
                let range_view_ty = self.units[alloc_id.0]
                    .content
                    .values
                    .keys()
                    .map(|(t, _)| *t)
                    .find(
                        |t| matches!(t.kind(), TyKind::Adt(adt, _) if adt.did() == adt_def.did()),
                    );
                let Some(range_view_ty) = range_view_ty else {
                    return;
                };
                let Some(start) = self.load_value(alloc_id, range_view_ty, &[0]).cloned() else {
                    return;
                };
                let Some(end) = self.load_value(alloc_id, range_view_ty, &[1]).cloned() else {
                    return;
                };
                let dest_ty = self.body().local_decls[dest].ty;
                let zero = Int::from_u64(self.z3_ctx, 0);
                let one = Int::from_u64(self.z3_ctx, 1);
                let start_term = start.z3_term.clone();
                let end_term = end.z3_term.clone();
                // `None` when `start >= end`; the `Some` path therefore has `start < end`.
                let is_empty = start_term.ge(&end_term);
                let result_val = VmValue {
                    z3_term: is_empty.ite(&zero, &start_term),
                    ty: dest_ty,
                    provenance: None,
                    facts: ValueFacts::default(),
                    source: ValueSource::Discriminant(is_empty.ite(&zero, &one)),
                };
                self.set_local(dest, result_val);
                self.set_field_value(dest, vec![0], start.clone());
                let mut advanced = start;
                let next = Int::add(self.z3_ctx, &[&start_term, &one]);
                advanced.z3_term = is_empty.ite(&start_term, &next);
                self.store_value(alloc_id, range_view_ty, vec![0], advanced);
            }
        }
    }

    fn set_opaque_intrinsic_result(&mut self, dest: Local) {
        let ty = self.body().local_decls[dest].ty;
        let term = self.fresh_int("intrinsic_result");
        self.inline.opaque_intrinsic_terms.insert(term.clone());
        self.set_local(dest, VmValue::new(term, ty));
    }

    fn swap_memory_values(&mut self, left: &VmValue<'z3, 'tcx>, right: &VmValue<'z3, 'tcx>) -> bool {
        if let Some((left_local, right_local)) = self
            .find_local_by_address(&left.z3_term)
            .zip(self.find_local_by_address(&right.z3_term))
        {
            let ty = self.body().local_decls[left_local].ty;
            if left_local != right_local
                && ty == self.body().local_decls[right_local].ty
                && matches!(ty.kind(), TyKind::Int(_) | TyKind::Uint(_))
                && let Some((left_value, right_value)) = self
                    .local_value(left_local)
                    .cloned()
                    .zip(self.local_value(right_local).cloned())
            {
                self.set_local(left_local, right_value);
                self.set_local(right_local, left_value);
                return true;
            }
        }
        let Some((left_prov, right_prov)) =
            left.provenance.as_ref().zip(right.provenance.as_ref())
        else {
            return false;
        };
        let Some(ty) = mir_utils::pointee_ty(left.ty) else {
            return false;
        };
        if mir_utils::pointee_ty(right.ty) != Some(ty) {
            return false;
        }
        for id in [left_prov.alloc_id, right_prov.alloc_id] {
            let root = self.root_alloc(id);
            let alloc = self.alloc(id);
            if alloc.element_ty.as_ty() != Some(ty)
                && self.alloc(root).element_ty.as_ty() != Some(ty)
            {
                return false;
            }
            if !self.content(id).facts.initialized && !self.content(root).facts.initialized {
                return false;
            }
        }
        self.forget_memory_values(left_prov.alloc_id);
        self.forget_memory_values(right_prov.alloc_id);
        true
    }

    fn apply_vec_allocation(
        &mut self,
        dest: Local,
        count: &VmValue<'z3, 'tcx>,
        elem_size: u64,
        initialized_len: bool,
    ) {
        let dest_ty = self.body().local_decls[dest].ty;
        let elem_ty = crate::verify::call_summary::effects::vec_elem_ty(dest_ty);
        let elem_sz = if elem_size == 0 {
            self.size_sym(elem_ty.unwrap_or(dest_ty))
        } else {
            Int::from_u64(self.z3_ctx, elem_size)
        };
        let total = Int::mul(self.z3_ctx, &[&count.z3_term, &elem_sz]);
        let heap_align = elem_ty
            .map(|ty| self.align_sym(ty))
            .unwrap_or_else(|| Int::from_u64(self.z3_ctx, 1));
        let (alloc_id, base) = self.allocate_external(total, heap_align, elem_ty);
        if initialized_len {
            self.alloc_mut(alloc_id)
                .set_slice_len(count.z3_term.clone());
        }
        let dest_alloc_id = self.current_frame.local_alloc.get(&dest).copied();
        self.content_mut(alloc_id).facts.initialized = true;
        let vec_base = base.clone();
        let vec_count = count.z3_term.clone();
        self.set_local(
            dest,
            VmValue {
                z3_term: base,
                ty: dest_ty,
                provenance: dest_alloc_id.map(|stack_id| Provenance {
                    alloc_id: stack_id,
                    offset: Int::from_u64(self.z3_ctx, 0),
                    offset_kind: None,
                }),
                facts: ValueFacts {
                    non_null: true,
                    init: true,
                    in_bounds: true,
                    ..ValueFacts::default()
                },
                source: ValueSource::None,
            },
        );
        if let TyKind::Adt(adt_def, _) = dest_ty.kind()
            && api_classify::is_std_vec(adt_def.did())
        {
            let ptr_field = VmValue {
                z3_term: vec_base,
                ty: elem_ty.unwrap_or(dest_ty),
                provenance: Some(Provenance {
                    alloc_id,
                    offset: Int::from_u64(self.z3_ctx, 0),
                    offset_kind: None,
                }),
                facts: ValueFacts {
                    non_null: true,
                    init: true,
                    in_bounds: true,
                    ..ValueFacts::default()
                },
                source: ValueSource::None,
            };
            let len = if initialized_len {
                vec_count.clone()
            } else {
                Int::from_u64(self.z3_ctx, 0)
            };
            self.materialize_vec_fields(dest, ptr_field, vec_count, len);
        }
    }
}
